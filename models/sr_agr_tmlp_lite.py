import time
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .abstract import TabModel, check_dir
from .tmlp_variant_blocks import TMLPBackbone


class _SRAGRTMLPLite(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        beta_init_bias: float = -2.0,
        use_gate_reg: bool = False,
        use_corr_reg: bool = False,
        use_featmix: bool = False,
        use_hidmix: bool = False,
        use_sparse_reg: bool = False,
        use_balance_reg: bool = False,
        **backbone_config,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("sr_agr_tmlp_lite currently supports regression only")
        if d_numerical <= 0:
            raise NotImplementedError("sr_agr_tmlp_lite requires numerical features")
        if use_hidmix:
            raise NotImplementedError("sr_agr_tmlp_lite does not implement hidmix")

        self.backbone = TMLPBackbone(
            d_numerical=d_numerical,
            categories=categories,
            **backbone_config,
        )
        self.n_num_features = d_numerical
        self.use_gate_reg = use_gate_reg
        self.use_corr_reg = use_corr_reg
        self.use_featmix = use_featmix
        self.use_hidmix = use_hidmix
        self.use_sparse_reg = use_sparse_reg
        self.use_balance_reg = use_balance_reg

        d_token = self.backbone.d_token
        gate_hidden = max(d_numerical // 2, 16)
        beta_hidden = max(d_token // 4, 16)
        residual_hidden = max(d_token // 2, 16)

        # Baseline path: input gate + original TMLP-style backbone + base head.
        self.input_gate = nn.Sequential(
            nn.Linear(d_numerical, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, d_numerical),
        )
        self.base_head = nn.Linear(d_token, d_out)

        # Very light residual correction head.
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, d_out),
        )

        # Sample-wise soft fusion gate.
        self.beta_head = nn.Sequential(
            nn.LayerNorm(d_token + d_out + 3),
            nn.Linear(d_token + d_out + 3, beta_hidden),
            nn.ReLU(),
            nn.Linear(beta_hidden, 1),
        )
        self._init_heads(beta_init_bias)

    def _init_heads(self, beta_init_bias: float) -> None:
        gate_last = ty.cast(nn.Linear, self.input_gate[-1])
        nn.init.zeros_(gate_last.weight)
        nn.init.zeros_(gate_last.bias)

        residual_last = ty.cast(nn.Linear, self.residual_head[-1])
        nn.init.zeros_(residual_last.weight)
        nn.init.zeros_(residual_last.bias)

        beta_last = ty.cast(nn.Linear, self.beta_head[-1])
        nn.init.zeros_(beta_last.weight)
        nn.init.constant_(beta_last.bias, beta_init_bias)

    @staticmethod
    def _gate_stats(gate_scores: torch.Tensor) -> torch.Tensor:
        gate_mean = gate_scores.mean(dim=1, keepdim=True)
        gate_max = gate_scores.max(dim=1, keepdim=True).values
        gate_std = gate_scores.std(dim=1, unbiased=False, keepdim=True)
        return torch.cat([gate_mean, gate_max, gate_std], dim=1)

    def forward(self, x_num, x_cat, return_extras: bool = False):
        if x_num is None:
            raise NotImplementedError("sr_agr_tmlp_lite requires numerical features")

        # Baseline path.
        gate_scores = torch.sigmoid(self.input_gate(x_num))
        x_gated = x_num * gate_scores
        h_base, _ = self.backbone.encode_hidden(x_gated, x_cat)
        y_base = self.base_head(h_base)

        # Low-disturbance residual correction.
        delta_y = self.residual_head(h_base)

        # Sample-wise fusion gate.
        beta_input = torch.cat([h_base, y_base, self._gate_stats(gate_scores)], dim=1)
        beta = torch.sigmoid(self.beta_head(beta_input))
        prediction = y_base + beta * delta_y

        if not return_extras:
            return prediction.squeeze(-1), {}

        return prediction.squeeze(-1), {
            "y_base": y_base,
            "delta_y": delta_y,
            "beta": beta,
            "gate_scores": gate_scores,
        }


class SRAGRTMLPLite(TabModel):
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
            raise NotImplementedError("sr_agr_tmlp_lite does not support sparse gating options yet")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _SRAGRTMLPLite(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "sr_agr_tmlp_lite"
        self.device = torch.device(device)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.setdefault("beta_init_bias", -2.0)
        model_config.setdefault("use_gate_reg", False)
        model_config.setdefault("use_corr_reg", False)
        model_config.setdefault("use_featmix", False)
        model_config.setdefault("use_hidmix", False)
        model_config.setdefault("use_sparse_reg", False)
        model_config.setdefault("use_balance_reg", False)
        return model_config

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
            raise NotImplementedError("sr_agr_tmlp_lite currently supports regression only")
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
        training_args.setdefault("lambda_gate", 0.0)
        training_args.setdefault("lambda_corr", 0.0)
        training_args.setdefault("featmix_alpha", 0.4)
        self.training_config = training_args

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
            beta_vals = []
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

                loss = F.huber_loss(predictions, mixed_y, delta=training_args["huber_delta"])
                if self.model.use_gate_reg:
                    loss = loss + training_args["lambda_gate"] * extras["beta"].mean()
                if self.model.use_corr_reg:
                    loss = loss + training_args["lambda_corr"] * extras["delta_y"].pow(2).mean()

                start_time = time.time()
                loss.backward()
                backward_time = time.time() - start_time
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                tot_time += forward_time + backward_time
                tot_step += 1
                tot_loss += loss.detach().cpu().item()
                beta_vals.append(float(extras["beta"].detach().mean().cpu().item()))
                delta_vals.append(float(extras["delta_y"].detach().abs().mean().cpu().item()))

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
                        f"[sr_agr_lite] beta_mean={np.mean(beta_vals):.6g}"
                        f" | abs_delta_y={np.mean(delta_vals):.6g}"
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
