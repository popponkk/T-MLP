import json
import math
import time
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
from .ggpl_cgr_tmlp import GGPLNumericEmbedding


class SmallMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int = 1,
        last_bias_init: float = 0.0,
        last_weight_std: float = 1e-3,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(hidden_dim), 8)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init(last_bias_init, last_weight_std)

    def _init(self, last_bias_init: float, last_weight_std: float) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn_init.normal_(module.weight, mean=0.0, std=1e-3)
                nn_init.zeros_(module.bias)
        last = ty.cast(nn.Linear, self.net[-1])
        nn_init.normal_(last.weight, mean=0.0, std=last_weight_std)
        nn_init.constant_(last.bias, last_bias_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TabMTrunk(nn.Module):
    def __init__(
        self,
        d_in: int,
        categories: ty.Optional[ty.List[int]],
        trunk_hidden_dim: int = 256,
        trunk_num_layers: int = 3,
        trunk_dropout: float = 0.1,
        cat_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.n_categories = 0 if categories is None else len(categories)
        if categories is not None:
            d_in += len(categories) * cat_embedding_dim
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), cat_embedding_dim)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        else:
            self.category_embeddings = None
            self.category_offsets = None

        hidden_dim = max(int(trunk_hidden_dim), 32)
        num_layers = max(int(trunk_num_layers), 1)
        layers = []
        current_dim = d_in
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if trunk_dropout:
                layers.append(nn.Dropout(trunk_dropout))
            current_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.output_dim = hidden_dim
        self._init()

    def _init(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn_init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                nn_init.zeros_(module.bias)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        parts = [x_num]
        if x_cat is not None and self.category_embeddings is not None:
            parts.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(x_cat.size(0), -1)
            )
        x = torch.cat(parts, dim=-1)
        return self.trunk(x)


class MultiHeadBasePredictor(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, head_hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_heads = max(int(num_heads), 2)
        head_hidden_dim = max(int(head_hidden_dim), 8)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, 1),
            )
            for _ in range(self.num_heads)
        ])
        self._init()

    def _init(self) -> None:
        for head_idx, head in enumerate(self.heads):
            for module in head:
                if isinstance(module, nn.Linear):
                    nn_init.normal_(module.weight, mean=0.0, std=1e-3)
                    nn_init.zeros_(module.bias)
            last = ty.cast(nn.Linear, head[-1])
            nn_init.normal_(last.weight, mean=0.0, std=1e-3 * (1.0 + 0.05 * head_idx))
            nn_init.constant_(last.bias, 1e-4 * head_idx)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        y_heads = torch.stack([head(h).squeeze(-1) for head in self.heads], dim=1)
        y_base = y_heads.mean(dim=1, keepdim=True)
        head_std = y_heads.std(dim=1, unbiased=False, keepdim=True)
        return y_base, y_heads, head_std


class _GGPLTabMCGR(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        use_ggpl_frontend: bool = True,
        num_breakpoints: int = 8,
        emb_dim_per_feature: int = 4,
        breakpoint_init: str = "gbdt",
        breakpoint_fallback: str = "quantile",
        learnable_breakpoints: bool = True,
        use_numeric_proj: bool = True,
        numeric_proj_dim: int = 256,
        numeric_proj_scale: float = 0.1,
        trunk_hidden_dim: int = 256,
        trunk_num_layers: int = 3,
        trunk_dropout: float = 0.1,
        cat_embedding_dim: int = 16,
        num_heads: int = 4,
        head_hidden_dim: int = 32,
        prediction_reduce: str = "mean",
        use_cgr_refinement: bool = True,
        safe_hidden_dim: int = 64,
        spec_hidden_dim: int = 64,
        spec_output_scale: float = 0.1,
        gamma_init_bias: float = -2.5,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("ggpl_tabm_cgr currently supports scalar regression only")
        if d_numerical <= 0:
            raise NotImplementedError("ggpl_tabm_cgr requires numerical features")
        if prediction_reduce != "mean":
            raise NotImplementedError("ggpl_tabm_cgr currently supports mean prediction reduction only")
        if not use_cgr_refinement:
            raise ValueError("ggpl_tabm_cgr requires use_cgr_refinement=True")

        self.use_ggpl_frontend = bool(use_ggpl_frontend)
        self.use_numeric_proj = bool(use_numeric_proj)
        self.learnable_breakpoints = bool(learnable_breakpoints)
        self.spec_output_scale = float(spec_output_scale)
        self.use_featmix = False
        self.use_corr_reg = False
        self.use_hidmix = False
        self.breakpoint_init = breakpoint_init
        self.breakpoint_fallback = breakpoint_fallback

        self.numeric_embedding = GGPLNumericEmbedding(
            n_features=d_numerical,
            num_breakpoints=num_breakpoints,
            emb_dim_per_feature=emb_dim_per_feature,
            use_numeric_proj=False,
        )
        if not self.learnable_breakpoints:
            self.numeric_embedding.breakpoint_delta_raw.requires_grad_(False)

        flat_dim = d_numerical * emb_dim_per_feature
        if self.use_numeric_proj:
            self.numeric_proj = nn.Linear(flat_dim, numeric_proj_dim)
            nn_init.normal_(self.numeric_proj.weight, mean=0.0, std=numeric_proj_scale * 1e-2)
            nn_init.zeros_(self.numeric_proj.bias)
            trunk_input_dim = numeric_proj_dim
        else:
            self.numeric_proj = None
            trunk_input_dim = flat_dim

        self.trunk = TabMTrunk(
            d_in=trunk_input_dim,
            categories=categories,
            trunk_hidden_dim=trunk_hidden_dim,
            trunk_num_layers=trunk_num_layers,
            trunk_dropout=trunk_dropout,
            cat_embedding_dim=cat_embedding_dim,
        )
        self.base_predictor = MultiHeadBasePredictor(
            input_dim=self.trunk.output_dim,
            num_heads=num_heads,
            head_hidden_dim=head_hidden_dim,
        )
        self.safe_head = SmallMLPHead(
            input_dim=self.trunk.output_dim,
            hidden_dim=safe_hidden_dim,
            out_dim=1,
            last_weight_std=1e-4,
        )
        self.spec_head = SmallMLPHead(
            input_dim=self.trunk.output_dim + 2,
            hidden_dim=spec_hidden_dim,
            out_dim=1,
            last_weight_std=1e-4,
        )
        self.conf_gate = SmallMLPHead(
            input_dim=self.trunk.output_dim + 4,
            hidden_dim=spec_hidden_dim,
            out_dim=1,
            last_bias_init=gamma_init_bias,
            last_weight_std=1e-4,
        )

    def set_breakpoints(self, breakpoints: Tensor) -> None:
        self.numeric_embedding.set_breakpoints(breakpoints)

    def forward(self, x_num, x_cat, return_extras: bool = False):
        if x_num is None:
            raise NotImplementedError("ggpl_tabm_cgr requires numerical features")
        if self.use_ggpl_frontend:
            _, numeric_embeddings, breakpoints = self.numeric_embedding(x_num)
            x_emb = numeric_embeddings.reshape(x_num.shape[0], -1)
            x_front = self.numeric_proj(x_emb) if self.numeric_proj is not None else x_emb
        else:
            x_front = x_num
            numeric_embeddings = None
            breakpoints = self.numeric_embedding.breakpoints()

        h = self.trunk(x_front, x_cat)
        y_base, y_heads, head_std = self.base_predictor(h)

        delta_safe = self.safe_head(h)
        spec_input = torch.cat([h, y_base, head_std], dim=1)
        raw_delta_spec = self.spec_head(spec_input)
        delta_spec = self.spec_output_scale * torch.tanh(raw_delta_spec)

        conf_input = torch.cat([h, y_base, head_std, delta_safe.abs(), delta_spec.abs()], dim=1)
        gamma = torch.sigmoid(self.conf_gate(conf_input))
        y_hat = y_base + delta_safe + gamma * delta_spec

        if not return_extras:
            return y_hat.squeeze(-1), {}
        return y_hat.squeeze(-1), {
            "h": h,
            "y_base": y_base,
            "y_heads": y_heads,
            "head_std": head_std,
            "delta_safe": delta_safe,
            "delta_spec": delta_spec,
            "gamma": gamma,
            "x_front": x_front,
            "numeric_embeddings": numeric_embeddings,
            "breakpoints": breakpoints,
        }


class GGPLTabMCGR(TabModel):
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
            raise NotImplementedError("ggpl_tabm_cgr does not support sparse gating options")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _GGPLTabMCGR(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggpl_tabm_cgr"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get("breakpoint_fallback", "quantile")
        self.breakpoint_cache_dir = self.saved_model_config.get("breakpoint_cache_dir", "artifacts/ggpl_tabm_cgr_breakpoints")
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("base_model", None)
        model_config.pop("breakpoint_cache_dir", None)
        model_config.pop("head_diversity_type", None)
        model_config.setdefault("use_ggpl_frontend", True)
        model_config.setdefault("num_breakpoints", 8)
        model_config.setdefault("emb_dim_per_feature", 4)
        model_config.setdefault("breakpoint_init", "gbdt")
        model_config.setdefault("breakpoint_fallback", "quantile")
        model_config.setdefault("learnable_breakpoints", True)
        model_config.setdefault("use_numeric_proj", True)
        model_config.setdefault("numeric_proj_dim", 256)
        model_config.setdefault("numeric_proj_scale", 0.1)
        model_config.setdefault("trunk_hidden_dim", 256)
        model_config.setdefault("trunk_num_layers", 3)
        model_config.setdefault("trunk_dropout", 0.1)
        model_config.setdefault("cat_embedding_dim", 16)
        model_config.setdefault("num_heads", 4)
        model_config.setdefault("head_hidden_dim", 32)
        model_config.setdefault("prediction_reduce", "mean")
        model_config.setdefault("use_cgr_refinement", True)
        model_config.setdefault("safe_hidden_dim", 64)
        model_config.setdefault("spec_hidden_dim", 64)
        model_config.setdefault("spec_output_scale", 0.1)
        model_config.setdefault("gamma_init_bias", -2.5)
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
            print(f"[ggpl_tabm_cgr] GBDT threshold extraction failed, falling back: {err}")
        return thresholds

    def _fit_or_load_breakpoints(self, x_num: ty.Optional[torch.Tensor], ys: ty.Optional[torch.Tensor], save_path: str) -> None:
        if x_num is None:
            return
        cache_file = self._cache_file(save_path, x_num.shape[1])
        if cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            breakpoints = torch.tensor(payload["breakpoints"], dtype=torch.float32, device=self.device)
            self.model.set_breakpoints(breakpoints)
            print(f"[ggpl_tabm_cgr] loaded breakpoints: {cache_file}")
            return

        x_np = x_num.detach().cpu().numpy()
        y_np = ys.detach().cpu().numpy() if ys is not None else np.zeros(len(x_np), dtype="float32")
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
        self.model.set_breakpoints(torch.tensor(breakpoints, dtype=torch.float32, device=self.device))
        print(f"[ggpl_tabm_cgr] saved breakpoints: {cache_file}")

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
            raise NotImplementedError("ggpl_tabm_cgr currently supports regression only")
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
        training_args.setdefault("main_loss_type", "huber")
        training_args.setdefault("huber_delta", 1.0)
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
            head_std_vals = []
            for batch in train_loader:
                optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
                start_time = time.time()
                predictions, extras = self.model(x_num, x_cat, return_extras=True)
                forward_time = time.time() - start_time

                if training_args["main_loss_type"] == "mse":
                    loss = F.mse_loss(predictions, y)
                elif training_args["main_loss_type"] == "huber":
                    loss = F.huber_loss(predictions, y, delta=training_args["huber_delta"])
                else:
                    raise ValueError(f"Unsupported main_loss_type: {training_args['main_loss_type']}")

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
                head_std_vals.append(float(extras["head_std"].detach().mean().cpu().item()))

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
                        f"[ggpl_tabm_cgr] head_std={np.mean(head_std_vals):.6g}"
                        f" | gamma_mean={np.mean(gamma_vals):.6g}"
                        f" | abs_delta_safe={np.mean(safe_vals):.6g}"
                        f" | abs_delta_spec={np.mean(spec_vals):.6g}"
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
