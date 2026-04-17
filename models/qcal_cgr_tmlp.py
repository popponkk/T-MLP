import time
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from torch.utils.data import DataLoader

from .abstract import TabModel, check_dir
from .tmlp_variant_blocks import TMLPBackbone


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_ratio: float,
        out_dim: int = 1,
        last_bias_init: float = 0.0,
        zero_last: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(input_dim * hidden_ratio), 16)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init(last_bias_init, zero_last)

    def _init(self, last_bias_init: float, zero_last: bool) -> None:
        last = ty.cast(nn.Linear, self.net[-1])
        if zero_last:
            nn_init.zeros_(last.weight)
        else:
            nn_init.normal_(last.weight, mean=0.0, std=1e-3)
        nn_init.constant_(last.bias, last_bias_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TopFeatureSummary(nn.Module):
    def __init__(self, summary_dim: int) -> None:
        super().__init__()
        summary_dim = max(int(summary_dim), 4)
        self.proj = nn.Sequential(
            nn.Linear(1, summary_dim),
            nn.ReLU(),
            nn.Linear(summary_dim, summary_dim),
        )
        self._init()

    def _init(self) -> None:
        for module in self.proj:
            if isinstance(module, nn.Linear):
                nn_init.normal_(module.weight, mean=0.0, std=1e-3)
                nn_init.zeros_(module.bias)

    def forward(self, topk_xw: Tensor) -> Tensor:
        # Shared lightweight projection over selected weighted scalars, then mean pooling.
        tokens = self.proj(topk_xw.unsqueeze(-1))
        return tokens.mean(dim=1)


class SharedCalibrationStem(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 16,
        num_layers: int = 2,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(hidden_dim), 4)
        num_layers = max(int(num_layers), 1)
        layers = []
        last_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU()])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
        self.residual_scale = float(residual_scale)
        self._init()

    def _init(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn_init.normal_(module.weight, mean=0.0, std=1e-3)
                nn_init.zeros_(module.bias)
        # Start identity-like: x_cal = x_raw + tiny residual.
        last = ty.cast(nn.Linear, self.net[-1])
        nn_init.zeros_(last.weight)
        nn_init.zeros_(last.bias)

    def forward(self, x_views: Tensor, x_raw: Tensor) -> Tensor:
        residual = torch.tanh(self.net(x_views)).squeeze(-1)
        return x_raw + self.residual_scale * residual


class _QCALCGRTMLP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        residual_dropout_head: float = 0.0,
        gate_reg_weight_scale: float = 0.1,
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
        use_raw_view: bool = True,
        use_robust_view: bool = True,
        use_quantile_view: bool = True,
        calib_hidden_dim: int = 16,
        calib_num_layers: int = 2,
        calib_residual_scale: float = 0.1,
        **backbone_config,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("qcal_cgr_tmlp currently supports regression only")
        if d_numerical <= 0:
            raise NotImplementedError("qcal_cgr_tmlp requires numerical features")
        if use_hidmix:
            raise NotImplementedError("qcal_cgr_tmlp does not implement hidmix")

        # Keep the same AGR-style gated-input + TMLP backbone trunk.
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
        self.use_raw_view = bool(use_raw_view)
        self.use_robust_view = bool(use_robust_view)
        self.use_quantile_view = bool(use_quantile_view)

        d_token = self.backbone.d_token
        gate_hidden = max(d_numerical // 2, 16)

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
        self.calibration_stem = SharedCalibrationStem(
            input_dim=3,
            hidden_dim=calib_hidden_dim,
            num_layers=calib_num_layers,
            residual_scale=calib_residual_scale,
        )
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
        self.register_buffer("robust_center", torch.zeros(1, d_numerical))
        self.register_buffer("robust_scale", torch.ones(1, d_numerical))
        self.register_buffer("quantile_sorted", torch.empty(0, d_numerical))

    def _init_gate(self) -> None:
        gate_last = ty.cast(nn.Linear, self.input_gate[-1])
        nn_init.zeros_(gate_last.weight)
        nn_init.zeros_(gate_last.bias)

    def set_calibration_stats(
        self,
        robust_center: Tensor,
        robust_scale: Tensor,
        quantile_sorted: Tensor,
    ) -> None:
        self.robust_center = robust_center.detach().to(self.robust_center.device)
        self.robust_scale = robust_scale.detach().to(self.robust_scale.device)
        self.quantile_sorted = quantile_sorted.detach().to(self.robust_center.device)

    def _quantile_view(self, x_num: Tensor) -> Tensor:
        if self.quantile_sorted.numel() == 0:
            return torch.zeros_like(x_num)
        sorted_values = self.quantile_sorted.to(x_num.device)
        n = sorted_values.shape[0]
        if n <= 1:
            return torch.zeros_like(x_num)
        views = []
        for feature_idx in range(x_num.shape[1]):
            ranks = torch.searchsorted(
                sorted_values[:, feature_idx].contiguous(),
                x_num[:, feature_idx].contiguous(),
                right=False,
            )
            views.append((ranks.float() / float(n - 1)).clamp(0.0, 1.0))
        return torch.stack(views, dim=1)

    def _calibrate_input(self, x_num: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_raw = x_num if self.use_raw_view else torch.zeros_like(x_num)
        x_robust = (
            (x_num - self.robust_center.to(x_num.device)) / self.robust_scale.to(x_num.device)
            if self.use_robust_view
            else torch.zeros_like(x_num)
        )
        x_quantile = self._quantile_view(x_num) if self.use_quantile_view else torch.zeros_like(x_num)
        x_views = torch.stack([x_raw, x_robust, x_quantile], dim=-1)
        x_cal = self.calibration_stem(x_views, x_num)
        return x_cal, x_robust, x_quantile

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
            raise NotImplementedError("qcal_cgr_tmlp requires numerical features")

        x_cal, x_robust, x_quantile = self._calibrate_input(x_num)

        # Original CGR backend starts here and keeps the same correction math.
        g = torch.sigmoid(self.input_gate(x_cal))
        x_gated = x_cal * g
        h_base, _ = self.backbone.encode_hidden(x_gated, x_cat)
        y_base = self.base_head(h_base)

        gate_mean, gate_max, gate_std = self._gate_stats(g)
        delta_safe = self.safe_residual_head(h_base)

        topk_x, topk_g = self._select_topk(x_cal, g)
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

        # Confidence-gated correction: generic safe residual is always active.
        y_hat = y_base + delta_safe + gamma * delta_spec

        if not return_extras:
            return y_hat.squeeze(-1), {}

        return y_hat.squeeze(-1), {
            "y_base": y_base,
            "delta_safe": delta_safe,
            "delta_spec": delta_spec,
            "gamma": gamma,
            "z_top": z_top,
            "x_raw": x_num,
            "x_robust": x_robust,
            "x_quantile": x_quantile,
            "x_cal": x_cal,
            "gate": g,
            "topk_gate_mean": topk_g.mean(dim=1, keepdim=True),
            "gate_scores": g,
        }


class QCALCGRTMLP(TabModel):
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
            raise NotImplementedError("qcal_cgr_tmlp does not support sparse gating options yet")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _QCALCGRTMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "qcal_cgr_tmlp"
        self.device = torch.device(device)
        self.quantile_view_type = self.saved_model_config.get("quantile_view_type", "rank_percentile")

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("base_model", None)
        model_config.pop("quantile_view_type", None)
        model_config.setdefault("residual_dropout_head", 0.0)
        model_config.setdefault("gate_reg_weight_scale", 0.1)
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
        model_config.setdefault("use_raw_view", True)
        model_config.setdefault("use_robust_view", True)
        model_config.setdefault("use_quantile_view", True)
        model_config.setdefault("calib_hidden_dim", 16)
        model_config.setdefault("calib_num_layers", 2)
        model_config.setdefault("calib_residual_scale", 0.1)
        return model_config

    def _fit_calibration_stats(self, x_num: ty.Optional[torch.Tensor]) -> None:
        if x_num is None:
            return
        if self.quantile_view_type != "rank_percentile":
            raise NotImplementedError(f"Unsupported quantile_view_type: {self.quantile_view_type}")
        center = x_num.median(dim=0, keepdim=True).values
        q75 = torch.quantile(x_num, 0.75, dim=0, keepdim=True)
        q25 = torch.quantile(x_num, 0.25, dim=0, keepdim=True)
        scale = torch.clamp(q75 - q25, min=1e-6)
        sorted_values = torch.sort(x_num, dim=0).values
        self.model.set_calibration_stats(center, scale, sorted_values)

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
            raise NotImplementedError("qcal_cgr_tmlp currently supports regression only")
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
        self._fit_calibration_stats(X_num)

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
                loss = loss + (
                    training_args["lambda_gate"]
                    * self.model.gate_reg_weight_scale
                    * gate_reg
                )
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
                        f"[qcal_cgr] gamma_mean={np.mean(gamma_vals):.6g}"
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
