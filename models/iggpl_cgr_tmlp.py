import time
import json
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from torch.utils.data import DataLoader

from .abstract import TabModel, check_dir
from .cgr_tmlp import ResidualMLP, TopFeatureSummary
from .ggpl_cgr_tmlp import GGPLNumericEmbedding
from .tmlp_variant_blocks import TMLPBackbone


class StableGGPLNumericEmbedding(GGPLNumericEmbedding):
    def __init__(
        self,
        n_features: int,
        num_breakpoints: int,
        emb_dim_per_feature: int,
        breakpoint_dropout: float = 0.05,
    ) -> None:
        super().__init__(
            n_features=n_features,
            num_breakpoints=num_breakpoints,
            emb_dim_per_feature=emb_dim_per_feature,
            use_numeric_proj=False,
        )
        self.breakpoint_dropout = float(breakpoint_dropout)

    def forward(self, x_num: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        breakpoints = self.breakpoints().to(x_num.device)
        hinge_basis = F.relu(x_num.unsqueeze(-1) - breakpoints.unsqueeze(0))
        if self.training and self.breakpoint_dropout > 0.0:
            keep = torch.rand(
                (1, hinge_basis.shape[1], hinge_basis.shape[2]),
                device=hinge_basis.device,
            ) >= self.breakpoint_dropout
            hinge_basis = hinge_basis * keep / max(1e-6, 1.0 - self.breakpoint_dropout)
        basis = torch.cat([x_num.unsqueeze(-1), hinge_basis], dim=-1)
        embeddings = torch.einsum("bdf,dfe->bde", basis, self.basis_weight) + self.basis_bias
        return x_num, embeddings, breakpoints


class IdentityResidualAdapter(nn.Module):
    def __init__(
        self,
        n_features: int,
        emb_dim_per_feature: int,
        delta_scale: float = 0.1,
        alpha_scale: float = 0.1,
        alpha_init: float = 0.05,
        hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.alpha_scale = float(alpha_scale)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim > 0:
            self.residual_out = nn.Sequential(
                nn.Linear(emb_dim_per_feature, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            self.residual_out = nn.Linear(emb_dim_per_feature, 1)
        self.alpha_param = nn.Parameter(torch.empty(n_features))
        self._init(alpha_init)

    def _init(self, alpha_init: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn_init.normal_(module.weight, mean=0.0, std=1e-3)
                nn_init.zeros_(module.bias)
        ratio = min(max(alpha_init / max(self.alpha_scale, 1e-6), 1e-4), 1.0 - 1e-4)
        init_param = float(np.log(ratio / (1.0 - ratio)))
        nn_init.constant_(self.alpha_param, init_param)

    def forward(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        raw_delta = self.residual_out(embeddings).squeeze(-1)
        delta_x = self.delta_scale * torch.tanh(raw_delta)
        alpha = self.alpha_scale * torch.sigmoid(self.alpha_param).unsqueeze(0)
        return delta_x, alpha


class _IGGPLCGRTMLP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        residual_dropout_head: float = 0.0,
        gate_reg_weight_scale: float = 0.05,
        use_corr_reg: bool = False,
        use_featmix: bool = False,
        use_hidmix: bool = False,
        safe_hidden_ratio: float = 0.5,
        spec_hidden_ratio: float = 0.25,
        conf_hidden_ratio: float = 0.25,
        topk_ratio: float = 0.15,
        topk_min: int = 4,
        topk_max: int = 8,
        top_summary_dim: int = 16,
        gamma_init_bias: float = -2.0,
        use_ggpl_path: bool = True,
        num_breakpoints: int = 8,
        emb_dim_per_feature: int = 4,
        learnable_breakpoints: bool = True,
        breakpoint_dropout: float = 0.05,
        use_residual_adapter: bool = True,
        delta_scale: float = 0.1,
        alpha_scale: float = 0.1,
        alpha_init: float = 0.05,
        residual_head_hidden_dim: int = 0,
        **backbone_config,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("iggpl_cgr_tmlp currently supports regression only")
        if d_numerical <= 0:
            raise NotImplementedError("iggpl_cgr_tmlp requires numerical features")
        if use_hidmix:
            raise NotImplementedError("iggpl_cgr_tmlp does not implement hidmix")
        if not use_residual_adapter:
            raise ValueError("iggpl_cgr_tmlp requires use_residual_adapter=True")

        self.backbone = TMLPBackbone(
            d_numerical=d_numerical,
            categories=categories,
            **backbone_config,
        )
        self.n_num_features = d_numerical
        self.use_gate_reg = True
        self.gate_reg_weight_scale = gate_reg_weight_scale
        self.use_corr_reg = use_corr_reg
        self.use_featmix = use_featmix
        self.use_hidmix = use_hidmix
        self.topk_ratio = topk_ratio
        self.topk_min = int(topk_min)
        self.topk_max = int(topk_max)
        self.use_ggpl_path = bool(use_ggpl_path)
        self.learnable_breakpoints = bool(learnable_breakpoints)

        d_token = self.backbone.d_token
        gate_hidden = max(d_numerical // 2, 16)

        self.numeric_embedding = StableGGPLNumericEmbedding(
            n_features=d_numerical,
            num_breakpoints=num_breakpoints,
            emb_dim_per_feature=emb_dim_per_feature,
            breakpoint_dropout=breakpoint_dropout,
        )
        if not self.learnable_breakpoints:
            self.numeric_embedding.breakpoint_delta_raw.requires_grad_(False)
        self.residual_adapter = IdentityResidualAdapter(
            n_features=d_numerical,
            emb_dim_per_feature=emb_dim_per_feature,
            delta_scale=delta_scale,
            alpha_scale=alpha_scale,
            alpha_init=alpha_init,
            hidden_dim=residual_head_hidden_dim,
        )

        self.input_gate = nn.Sequential(
            nn.Linear(d_numerical, gate_hidden),
            nn.ReLU(),
            nn.Dropout(residual_dropout_head),
            nn.Linear(gate_hidden, d_numerical),
        )
        self.base_head = nn.Linear(d_token, d_out)
        self.safe_residual_head = ResidualMLP(
            d_token,
            hidden_ratio=safe_hidden_ratio,
            out_dim=d_out,
            zero_last=True,
        )
        self.top_summary = TopFeatureSummary(top_summary_dim)
        self.spec_head = ResidualMLP(
            d_token + top_summary_dim,
            hidden_ratio=spec_hidden_ratio,
            out_dim=1,
            zero_last=False,
        )
        self.conf_gate = ResidualMLP(
            d_token + d_out + 3 + top_summary_dim + d_out + d_out,
            hidden_ratio=conf_hidden_ratio,
            out_dim=1,
            last_bias_init=gamma_init_bias,
            zero_last=False,
        )
        self._init_gate()

    def _init_gate(self) -> None:
        gate_last = ty.cast(nn.Linear, self.input_gate[-1])
        nn_init.zeros_(gate_last.weight)
        nn_init.zeros_(gate_last.bias)

    @staticmethod
    def _gate_stats(g: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        gate_mean = g.mean(dim=1, keepdim=True)
        gate_max = g.max(dim=1, keepdim=True).values
        gate_std = g.std(dim=1, unbiased=False, keepdim=True)
        return gate_mean, gate_max, gate_std

    def _select_topk(self, x_num: Tensor, g: Tensor) -> tuple[Tensor, Tensor]:
        n_features = x_num.shape[1]
        k = min(n_features, max(self.topk_min, min(self.topk_max, round(self.topk_ratio * n_features))))
        topk_idx = torch.topk(g, k=k, dim=1).indices
        topk_x = torch.gather(x_num, dim=1, index=topk_idx)
        topk_g = torch.gather(g, dim=1, index=topk_idx)
        return topk_x, topk_g

    def forward(self, x_num, x_cat, return_extras: bool = False):
        if x_num is None:
            raise NotImplementedError("iggpl_cgr_tmlp requires numerical features")

        x_raw = x_num
        if self.use_ggpl_path:
            _, numeric_embeddings, breakpoints = self.numeric_embedding(x_raw)
            delta_x, alpha = self.residual_adapter(numeric_embeddings)
            x_in = x_raw + alpha * delta_x
        else:
            numeric_embeddings = None
            breakpoints = self.numeric_embedding.breakpoints()
            delta_x = torch.zeros_like(x_raw)
            alpha = torch.zeros((1, x_raw.shape[1]), device=x_raw.device, dtype=x_raw.dtype)
            x_in = x_raw

        g = torch.sigmoid(self.input_gate(x_in))
        x_gated = x_in * g
        h_base, _ = self.backbone.encode_hidden(x_gated, x_cat)
        y_base = self.base_head(h_base)

        gate_mean, gate_max, gate_std = self._gate_stats(g)
        delta_safe = self.safe_residual_head(h_base)

        topk_x, topk_g = self._select_topk(x_in, g)
        topk_w = topk_g / (topk_g.sum(dim=1, keepdim=True) + 1e-6)
        z_top = self.top_summary(topk_x * topk_w)

        delta_spec = self.spec_head(torch.cat([h_base, z_top], dim=1))
        conf_input = torch.cat(
            [
                h_base,
                y_base,
                gate_mean,
                gate_max,
                gate_std,
                z_top,
                delta_safe.abs(),
                delta_spec.abs(),
            ],
            dim=1,
        )
        gamma = torch.sigmoid(self.conf_gate(conf_input))
        y_hat = y_base + delta_safe + gamma * delta_spec

        if not return_extras:
            return y_hat.squeeze(-1), {}

        return y_hat.squeeze(-1), {
            "y_base": y_base,
            "delta_safe": delta_safe,
            "delta_spec": delta_spec,
            "gamma": gamma,
            "z_top": z_top,
            "x_raw": x_raw,
            "delta_x": delta_x,
            "alpha": alpha,
            "x_in": x_in,
            "numeric_embeddings": numeric_embeddings,
            "breakpoints": breakpoints,
            "topk_gate_mean": topk_g.mean(dim=1, keepdim=True),
            "gate_scores": g,
        }


class IGGPLCGRTMLP(TabModel):
    def __init__(
        self,
        model_config: dict,
        n_num_features: int,
        categories: ty.Optional[ty.List[int]],
        n_labels: int,
        device: ty.Union[str, torch.device] = "cuda",
        feat_gate: ty.Optional[str] = None,
        pruning: ty.Optional[str] = None,
        dataset=None,
    ):
        if feat_gate or pruning:
            raise NotImplementedError("iggpl_cgr_tmlp does not support sparse gating options yet")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _IGGPLCGRTMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "iggpl_cgr_tmlp"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get("breakpoint_fallback", "quantile")
        self.breakpoint_cache_dir = self.saved_model_config.get("breakpoint_cache_dir", "artifacts/iggpl_breakpoints")
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("base_model", None)
        model_config.pop("breakpoint_init", None)
        model_config.pop("breakpoint_fallback", None)
        model_config.pop("breakpoint_cache_dir", None)
        model_config.setdefault("residual_dropout_head", 0.0)
        model_config.setdefault("gate_reg_weight_scale", 0.05)
        model_config.setdefault("use_corr_reg", False)
        model_config.setdefault("use_featmix", False)
        model_config.setdefault("use_hidmix", False)
        model_config.setdefault("safe_hidden_ratio", 0.5)
        model_config.setdefault("spec_hidden_ratio", 0.25)
        model_config.setdefault("conf_hidden_ratio", 0.25)
        model_config.setdefault("topk_ratio", 0.15)
        model_config.setdefault("topk_min", 4)
        model_config.setdefault("topk_max", 8)
        model_config.setdefault("top_summary_dim", 16)
        model_config.setdefault("gamma_init_bias", -2.0)
        model_config.setdefault("use_ggpl_path", True)
        model_config.setdefault("num_breakpoints", 8)
        model_config.setdefault("emb_dim_per_feature", 4)
        model_config.setdefault("learnable_breakpoints", True)
        model_config.setdefault("breakpoint_dropout", 0.05)
        model_config.setdefault("use_residual_adapter", True)
        model_config.setdefault("delta_scale", 0.1)
        model_config.setdefault("alpha_scale", 0.1)
        model_config.setdefault("alpha_init", 0.05)
        model_config.setdefault("residual_head_hidden_dim", 0)
        return model_config

    @staticmethod
    def _quantile_breakpoints(x_np: np.ndarray, num_breakpoints: int) -> np.ndarray:
        qs = np.linspace(0.1, 0.9, num_breakpoints)
        return np.quantile(x_np, qs, axis=0).T.astype("float32")

    @staticmethod
    def _even_breakpoints(x_np: np.ndarray, num_breakpoints: int) -> np.ndarray:
        mins = np.nanmin(x_np, axis=0)
        maxs = np.nanmax(x_np, axis=0)
        qs = np.linspace(0.1, 0.9, num_breakpoints)
        return np.stack([mins + q * (maxs - mins) for q in qs], axis=1).astype("float32")

    def _cache_file(self, save_path: str, n_features: int) -> Path:
        dataset_name = Path(save_path).name
        return Path(self.breakpoint_cache_dir) / dataset_name / f"breakpoints_f{n_features}_b{self.num_breakpoints}.json"

    def _extract_gbdt_thresholds(self, x_np: np.ndarray, y_np: np.ndarray) -> list[list[float]]:
        thresholds = [[] for _ in range(x_np.shape[1])]
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            gbdt = GradientBoostingRegressor(
                n_estimators=64,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            gbdt.fit(x_np, y_np.reshape(-1))
            for estimator in gbdt.estimators_.ravel():
                tree = estimator.tree_
                for feature_idx, threshold in zip(tree.feature, tree.threshold):
                    if feature_idx >= 0 and np.isfinite(threshold):
                        thresholds[int(feature_idx)].append(float(threshold))
        except Exception as err:
            print(f"[iggpl] GBDT threshold extraction failed, falling back: {err}")
        return thresholds

    def _fit_or_load_breakpoints(self, x_num: torch.Tensor, ys: torch.Tensor, save_path: str) -> None:
        cache_file = self._cache_file(save_path, x_num.shape[1])
        if cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            breakpoints = torch.tensor(payload["breakpoints"], dtype=torch.float32, device=self.device)
            self.model.numeric_embedding.set_breakpoints(breakpoints)
            print(f"[iggpl] loaded breakpoints: {cache_file}")
            return

        x_np = x_num.detach().cpu().numpy()
        y_np = ys.detach().cpu().numpy()
        fallback = (
            self._quantile_breakpoints(x_np, self.num_breakpoints)
            if self.breakpoint_fallback == "quantile"
            else self._even_breakpoints(x_np, self.num_breakpoints)
        )
        breakpoints = fallback.copy()

        if self.breakpoint_init == "gbdt":
            thresholds = self._extract_gbdt_thresholds(x_np, y_np)
            for feature_idx, values in enumerate(thresholds):
                unique = np.array(sorted(set(values)), dtype="float32")
                if len(unique) >= self.num_breakpoints:
                    pick = np.linspace(0, len(unique) - 1, self.num_breakpoints).round().astype(int)
                    breakpoints[feature_idx] = unique[pick]
                elif len(unique) > 0:
                    merged = np.array(sorted(set(unique.tolist() + fallback[feature_idx].tolist())), dtype="float32")
                    pick = np.linspace(0, len(merged) - 1, self.num_breakpoints).round().astype(int)
                    breakpoints[feature_idx] = merged[pick]
        elif self.breakpoint_init not in ["quantile", "even"]:
            raise ValueError(f"Unsupported breakpoint_init: {self.breakpoint_init}")

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "breakpoint_init": self.breakpoint_init,
                    "breakpoint_fallback": self.breakpoint_fallback,
                    "num_breakpoints": self.num_breakpoints,
                    "breakpoints": breakpoints.tolist(),
                },
                f,
                indent=2,
            )
        self.model.numeric_embedding.set_breakpoints(torch.tensor(breakpoints, dtype=torch.float32, device=self.device))
        print(f"[iggpl] saved breakpoints: {cache_file}")

    @staticmethod
    def _apply_featmix(x_num, y, enabled: bool, alpha: float = 0.4):
        if (not enabled) or x_num is None or len(x_num) <= 1:
            return x_num, y
        lam = np.random.beta(alpha, alpha)
        perm = torch.randperm(len(x_num), device=x_num.device)
        mixed_x = lam * x_num + (1.0 - lam) * x_num[perm]
        mixed_y = lam * y + (1.0 - lam) * y[perm]
        return mixed_x, mixed_y

    def fit(
        self,
        train_loader: ty.Optional[ty.Tuple[DataLoader, int]] = None,
        X_num: ty.Optional[torch.Tensor] = None,
        X_cat: ty.Optional[torch.Tensor] = None,
        ys: ty.Optional[torch.Tensor] = None,
        ids: ty.Optional[torch.Tensor] = None,
        y_std: ty.Optional[float] = None,
        eval_set: ty.Tuple[torch.Tensor, np.ndarray] = None,
        patience: int = 0,
        task: str = None,
        training_args: dict = None,
        meta_args: ty.Optional[dict] = None,
    ):
        if task != "regression":
            raise NotImplementedError("iggpl_cgr_tmlp currently supports regression only")
        if meta_args is None:
            meta_args = {}
        meta_args.setdefault("save_path", f"results/{self.base_name}")
        meta_args.setdefault("log_every_n_epochs", 50)
        check_dir(meta_args["save_path"])
        self.meta_config = meta_args

        if training_args is None:
            training_args = {}
        training_args.setdefault("optimizer", "adamw")
        training_args.setdefault("batch_size", 64)
        training_args.setdefault("max_epochs", 10000)
        training_args.setdefault("patience", patience)
        training_args.setdefault("save_frequency", "epoch")
        training_args.setdefault("huber_delta", 1.0)
        training_args.setdefault("main_loss_type", "huber")
        training_args.setdefault("lambda_gate", 1e-3)
        training_args.setdefault("lambda_corr", 0.0)
        training_args.setdefault("featmix_alpha", 0.4)
        self.training_config = training_args
        self._fit_or_load_breakpoints(X_num, ys, meta_args["save_path"])

        optimizer, scheduler = TabModel.make_optimizer(self.model, training_args)
        if train_loader is not None:
            train_loader, placeholders = train_loader
            training_args["batch_size"] = train_loader.batch_size
        else:
            train_loader, placeholders = TabModel.prepare_tensor_loader(
                X_num=X_num,
                X_cat=X_cat,
                ys=ys,
                ids=ids,
                batch_size=training_args["batch_size"],
                shuffle=True,
            )

        if eval_set is not None:
            dev_loader = TabModel.prepare_tensor_loader(
                X_num=eval_set[0][0], X_cat=eval_set[0][1], ys=eval_set[0][2], ids=eval_set[0][3],
                batch_size=training_args["batch_size"],
            )
            test_loader = None
            if len(eval_set) == 2:
                test_loader = TabModel.prepare_tensor_loader(
                    X_num=eval_set[1][0], X_cat=eval_set[1][1], ys=eval_set[1][2], ids=eval_set[1][3],
                    batch_size=training_args["batch_size"],
                )
        else:
            dev_loader, test_loader = None, None

        steps_per_epoch = len(train_loader)
        tot_step, tot_time = 0, 0.0
        for _ in range(training_args["max_epochs"]):
            self.model.train()
            tot_loss = 0.0
            gamma_vals = []
            safe_vals = []
            spec_vals = []
            z_top_vals = []
            alpha_vals = []
            delta_vals = []
            for batch in train_loader:
                optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
                mixed_num, mixed_y = self._apply_featmix(
                    x_num,
                    y,
                    enabled=self.model.use_featmix,
                    alpha=training_args["featmix_alpha"],
                )
                start_time = time.time()
                predictions, extras = self.model(mixed_num, x_cat, return_extras=True)
                forward_time = time.time() - start_time

                if training_args["main_loss_type"] == "mse":
                    loss = F.mse_loss(predictions, mixed_y)
                elif training_args["main_loss_type"] == "huber":
                    loss = F.huber_loss(predictions, mixed_y, delta=training_args["huber_delta"])
                else:
                    raise ValueError(f"Unsupported main_loss_type: {training_args['main_loss_type']}")
                gate_reg = extras["gate_scores"].mean()
                loss = loss + training_args["lambda_gate"] * self.model.gate_reg_weight_scale * gate_reg
                if self.model.use_corr_reg:
                    corr_term = extras["delta_safe"].pow(2).mean() + extras["delta_spec"].pow(2).mean()
                    loss = loss + training_args["lambda_corr"] * corr_term

                start_time = time.time()
                loss.backward()
                backward_time = time.time() - start_time
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                tot_time += forward_time + backward_time
                tot_step += 1
                tot_loss += loss.detach().cpu().item()
                gamma_vals.append(float(extras["gamma"].detach().mean().cpu().item()))
                safe_vals.append(float(extras["delta_safe"].detach().abs().mean().cpu().item()))
                spec_vals.append(float(extras["delta_spec"].detach().abs().mean().cpu().item()))
                z_top_vals.append(float(extras["z_top"].detach().norm(dim=1).mean().cpu().item()))
                alpha_vals.append(float(extras["alpha"].detach().mean().cpu().item()))
                delta_vals.append(float(extras["delta_x"].detach().abs().mean().cpu().item()))

            if training_args["save_frequency"] == "epoch":
                train_metrics = self.predict(
                    X_num=X_num,
                    X_cat=X_cat,
                    ys=ys,
                    y_std=y_std,
                    task=task,
                    return_metric=True,
                    return_loss=False,
                )[1]["metrics"]
                val_metrics = None
                if eval_set is not None:
                    val_metrics = self.predict(
                        X_num=eval_set[0][0],
                        X_cat=eval_set[0][1],
                        ys=eval_set[0][2],
                        y_std=y_std,
                        task=task,
                        return_metric=True,
                        return_loss=False,
                    )[1]["metrics"]
                self.append_log(
                    meta_args["save_path"],
                    (
                        f"[metrics] train_rmse={train_metrics['rmse']:.6g}"
                        f" | train_mae={train_metrics['mae']:.6g}"
                        f" | train_r2={train_metrics['r2']:.6g}"
                        + (
                            f" | val_rmse={val_metrics['rmse']:.6g}"
                            f" | val_mae={val_metrics['mae']:.6g}"
                            f" | val_r2={val_metrics['r2']:.6g}"
                            if val_metrics is not None else ""
                        )
                    ),
                )
                self.append_log(
                    meta_args["save_path"],
                    (
                        f"[iggpl_cgr] alpha_mean={np.mean(alpha_vals):.6g}"
                        f" | abs_delta_x={np.mean(delta_vals):.6g}"
                        f" | gamma_mean={np.mean(gamma_vals):.6g}"
                        f" | abs_delta_safe={np.mean(safe_vals):.6g}"
                        f" | abs_delta_spec={np.mean(spec_vals):.6g}"
                        f" | z_top_norm={np.mean(z_top_vals):.6g}"
                    ),
                )
                is_early_stop = self.save_evaluate_dnn(
                    tot_step, steps_per_epoch, tot_loss, tot_time,
                    task, training_args["patience"], meta_args["save_path"],
                    dev_loader, y_std, test_loader=test_loader,
                )
                if is_early_stop:
                    self.save(meta_args["save_path"])
                    self.load_best_dnn(meta_args["save_path"])
                    return
        self.save(meta_args["save_path"])
        self.load_best_dnn(meta_args["save_path"])

    def predict(
        self,
        dev_loader: ty.Optional[ty.Tuple[DataLoader, int]] = None,
        X_num: ty.Optional[torch.Tensor] = None,
        X_cat: ty.Optional[torch.Tensor] = None,
        ys: ty.Optional[torch.Tensor] = None,
        ids: ty.Optional[torch.Tensor] = None,
        y_std: ty.Optional[float] = None,
        task: str = None,
        return_probs: bool = True,
        return_metric: bool = False,
        return_loss: bool = False,
        meta_args: ty.Optional[dict] = None,
    ):
        def inference_step(model, x_num, x_cat):
            start = time.time()
            predictions, _ = model(x_num, x_cat, return_extras=False)
            used = time.time() - start
            return predictions, used

        return self.dnn_predict(
            dnn_predict_func=inference_step,
            dev_loader=dev_loader,
            X_num=X_num,
            X_cat=X_cat,
            ys=ys,
            ids=ids,
            y_std=y_std,
            task=task,
            return_probs=return_probs,
            return_metric=return_metric,
            return_loss=return_loss,
            meta_args=meta_args,
        )

    def save(self, output_dir):
        check_dir(output_dir)
        self.save_pt_model(output_dir)
        self.save_history(output_dir)
        self.save_config(output_dir)
