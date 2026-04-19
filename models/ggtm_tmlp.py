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
from .ggpl_cgr_tmlp import GGPLNumericEmbedding
from .tmlp_variant_blocks import TMLPBackbone


class MultiHeadPrediction(nn.Module):
    def __init__(
        self,
        d_token: int,
        d_out: int,
        num_heads: int = 4,
        head_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("ggtm_tmlp currently supports scalar regression only")
        self.num_heads = int(num_heads)
        head_hidden_dim = max(int(head_hidden_dim), 8)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_token),
                nn.Linear(d_token, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, d_out),
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

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        y_heads = torch.stack([head(h).squeeze(-1) for head in self.heads], dim=1)
        y_hat = y_heads.mean(dim=1)
        return y_hat, y_heads


class PairwiseRankHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        hidden_dim = max(int(hidden_dim), 8)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init()

    def _init(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn_init.normal_(module.weight, mean=0.0, std=1e-3)
                nn_init.zeros_(module.bias)

    def forward(self, pair_feat: Tensor) -> Tensor:
        return self.net(pair_feat).squeeze(-1)


class _GGTMTMLP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        use_ggpl_frontend: bool = True,
        num_breakpoints: int = 8,
        emb_dim_per_feature: int = 4,
        use_numeric_proj: bool = True,
        numeric_proj_scale: float = 0.1,
        learnable_breakpoints: bool = True,
        num_heads: int = 4,
        head_hidden_dim: int = 32,
        rank_hidden_dim: int = 64,
        **backbone_config,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("ggtm_tmlp currently supports regression only")
        if d_numerical <= 0:
            raise NotImplementedError("ggtm_tmlp requires numerical features")

        self.use_ggpl_frontend = bool(use_ggpl_frontend)
        self.use_numeric_proj = bool(use_numeric_proj)
        self.learnable_breakpoints = bool(learnable_breakpoints)
        self.use_featmix = False
        self.use_corr_reg = False
        self.use_hidmix = False

        self.numeric_embedding = GGPLNumericEmbedding(
            n_features=d_numerical,
            num_breakpoints=num_breakpoints,
            emb_dim_per_feature=emb_dim_per_feature,
            use_numeric_proj=use_numeric_proj,
            proj_scale=numeric_proj_scale,
        )
        if not self.learnable_breakpoints:
            self.numeric_embedding.breakpoint_delta_raw.requires_grad_(False)

        trunk_numerical = d_numerical if (not self.use_ggpl_frontend or use_numeric_proj) else d_numerical * emb_dim_per_feature
        self.backbone = TMLPBackbone(
            d_numerical=trunk_numerical,
            categories=categories,
            **backbone_config,
        )
        d_token = self.backbone.d_token
        self.prediction_heads = MultiHeadPrediction(
            d_token=d_token,
            d_out=d_out,
            num_heads=num_heads,
            head_hidden_dim=head_hidden_dim,
        )
        self.rank_head = PairwiseRankHead(4 * d_token, hidden_dim=rank_hidden_dim)

    def set_breakpoints(self, breakpoints: Tensor) -> None:
        self.numeric_embedding.set_breakpoints(breakpoints)

    def forward(self, x_num, x_cat, return_extras: bool = False):
        if x_num is None:
            raise NotImplementedError("ggtm_tmlp requires numerical features")
        if self.use_ggpl_frontend:
            x_front, numeric_embeddings, breakpoints = self.numeric_embedding(x_num)
        else:
            x_front = x_num
            numeric_embeddings = None
            breakpoints = self.numeric_embedding.breakpoints()

        h, token_states = self.backbone.encode_hidden(x_front, x_cat)
        y_hat, y_heads = self.prediction_heads(h)

        if not return_extras:
            return y_hat, {}
        return y_hat, {
            "h_enc": h,
            "token_states": token_states,
            "y_heads": y_heads,
            "x_front": x_front,
            "numeric_embeddings": numeric_embeddings,
            "breakpoints": breakpoints,
        }


class GGTMTMLP(TabModel):
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
            raise NotImplementedError("ggtm_tmlp does not support sparse gating options yet")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _GGTMTMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "ggtm_tmlp"
        self.device = torch.device(device)
        self.breakpoint_init = self.saved_model_config.get("breakpoint_init", "gbdt")
        self.breakpoint_fallback = self.saved_model_config.get("breakpoint_fallback", "quantile")
        self.breakpoint_cache_dir = self.saved_model_config.get("breakpoint_cache_dir", "artifacts/ggtm_breakpoints")
        self.num_breakpoints = int(self.saved_model_config.get("num_breakpoints", 8))

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("breakpoint_init", None)
        model_config.pop("breakpoint_fallback", None)
        model_config.pop("breakpoint_cache_dir", None)
        model_config.pop("head_diversity_type", None)
        model_config.pop("prediction_reduce", None)
        model_config.setdefault("token_bias", True)
        model_config.setdefault("n_layers", 1)
        model_config.setdefault("d_token", 1024)
        model_config.setdefault("d_ffn_factor", 0.66)
        model_config.setdefault("ffn_dropout", None)
        model_config.setdefault("residual_dropout", 0.1)
        model_config.setdefault("use_ggpl_frontend", True)
        model_config.setdefault("num_breakpoints", 8)
        model_config.setdefault("emb_dim_per_feature", 4)
        model_config.setdefault("use_numeric_proj", True)
        model_config.setdefault("numeric_proj_scale", 0.1)
        model_config.setdefault("learnable_breakpoints", True)
        model_config.setdefault("num_heads", 4)
        model_config.setdefault("head_hidden_dim", 32)
        model_config.setdefault("rank_hidden_dim", 64)
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
            print(f"[ggtm] GBDT threshold extraction failed, falling back: {err}")
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
            print(f"[ggtm] loaded breakpoints: {cache_file}")
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
        print(f"[ggtm] saved breakpoints: {cache_file}")

    @staticmethod
    def _resolve_pairs_per_batch(value, batch_size: int) -> int:
        if isinstance(value, str):
            if value == "batch_size":
                return batch_size
            if value == "2x_batch_size":
                return 2 * batch_size
            return int(value)
        return int(value)

    def _rank_aux_loss(self, h_enc: Tensor, y: Tensor, training_args: dict) -> Tensor:
        if not training_args.get("use_rank_aux", True):
            return h_enc.new_tensor(0.0)
        batch_size = h_enc.shape[0]
        if batch_size <= 1:
            return h_enc.new_tensor(0.0)
        n_pairs = self._resolve_pairs_per_batch(training_args.get("pairs_per_batch", "batch_size"), batch_size)
        n_pairs = max(1, n_pairs)
        idx_i = torch.randint(0, batch_size, (n_pairs,), device=h_enc.device)
        offset = torch.randint(1, batch_size, (n_pairs,), device=h_enc.device)
        idx_j = (idx_i + offset) % batch_size
        h_i = h_enc[idx_i]
        h_j = h_enc[idx_j]
        pair_feat = torch.cat([h_i, h_j, h_i - h_j, (h_i - h_j).abs()], dim=1)
        rank_logit = self.model.rank_head(pair_feat)
        y_flat = y.reshape(-1)
        rank_label = (y_flat[idx_i] > y_flat[idx_j]).float()
        return F.binary_cross_entropy_with_logits(rank_logit, rank_label)

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
            raise NotImplementedError("ggtm_tmlp currently supports regression only")
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
        training_args.setdefault("featmix_alpha", 0.4)
        training_args.setdefault("use_rank_aux", True)
        training_args.setdefault("lambda_rank", 0.05)
        training_args.setdefault("pairs_per_batch", "batch_size")
        self.training_config = training_args
        if self.model.use_ggpl_frontend:
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
            rank_vals = []
            head_std_vals = []
            for batch in train_loader:
                optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
                mixed_num, mixed_y = self._apply_featmix(x_num, y, enabled=False, alpha=training_args["featmix_alpha"])
                start_time = time.time()
                predictions, extras = self.model(mixed_num, x_cat, return_extras=True)
                forward_time = time.time() - start_time

                if training_args["main_loss_type"] == "mse":
                    main_loss = F.mse_loss(predictions, mixed_y)
                elif training_args["main_loss_type"] == "huber":
                    main_loss = F.huber_loss(predictions, mixed_y, delta=training_args["huber_delta"])
                else:
                    raise ValueError(f"Unsupported main_loss_type: {training_args['main_loss_type']}")
                rank_loss = self._rank_aux_loss(extras["h_enc"], mixed_y, training_args)
                loss = main_loss + training_args["lambda_rank"] * rank_loss

                start_time = time.time()
                loss.backward()
                backward_time = time.time() - start_time
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                tot_time += forward_time + backward_time
                tot_step += 1
                tot_loss += loss.detach().cpu().item()
                rank_vals.append(float(rank_loss.detach().cpu().item()))
                head_std_vals.append(float(extras["y_heads"].detach().std(dim=1).mean().cpu().item()))

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
                        f"[ggtm] rank_loss={np.mean(rank_vals):.6g}"
                        f" | head_std={np.mean(head_std_vals):.6g}"
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
