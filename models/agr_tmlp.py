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


class AGRResidualHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        mid_dim = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )
        self._init_small()

    def _init_small(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn_init.normal_(module.weight, mean=0.0, std=1e-3)
                nn_init.zeros_(module.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.net(hidden)


class AGRErrorHead(AGRResidualHead):
    def forward(self, hidden: Tensor) -> Tensor:
        return F.softplus(super().forward(hidden))


class AGRGate(nn.Module):
    def __init__(self, hidden_dim: int, use_error_feature: bool) -> None:
        super().__init__()
        self.use_error_feature = use_error_feature
        gate_in = hidden_dim + 1 if use_error_feature else hidden_dim
        gate_hidden = max(hidden_dim // 4, 16)
        self.net = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        self._init_small()

    def _init_small(self) -> None:
        first = ty.cast(nn.Linear, self.net[0])
        last = ty.cast(nn.Linear, self.net[-1])
        nn_init.normal_(first.weight, mean=0.0, std=1e-3)
        nn_init.zeros_(first.bias)
        nn_init.normal_(last.weight, mean=0.0, std=1e-3)
        nn_init.constant_(last.bias, -4.0)

    def forward(self, hidden: Tensor, predicted_error: Tensor) -> Tensor:
        gate_input = hidden
        if self.use_error_feature:
            gate_input = torch.cat([hidden, predicted_error], dim=1)
        return torch.sigmoid(self.net(gate_input))


class _AGRTMLP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        residual_dropout_head: float = 0.1,
        use_error_head: bool = True,
        gate_from_h_only: bool = False,
        **backbone_config,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("AGR-TMLP currently supports regression only")

        self.backbone = TMLPBackbone(
            d_numerical=d_numerical,
            categories=categories,
            **backbone_config,
        )
        self.base_head = nn.Linear(self.backbone.d_token, d_out)
        self.residual_head = AGRResidualHead(self.backbone.d_token, residual_dropout_head)
        self.use_error_head = use_error_head
        self.error_head = AGRErrorHead(self.backbone.d_token, residual_dropout_head)
        self.gate = AGRGate(self.backbone.d_token, use_error_feature=not gate_from_h_only)

    def set_backbone_trainable(
        self,
        trainable: bool,
        last_layer_only: bool = False,
    ) -> None:
        modules = [self.backbone.tokenizer, self.backbone.normalization]
        if last_layer_only and len(self.backbone.layers) > 0:
            modules.append(self.backbone.layers[-1])
        else:
            modules.extend(self.backbone.layers)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = trainable

    def forward(self, x_num, x_cat, return_extras: bool = False):
        hidden, _ = self.backbone.encode_hidden(x_num, x_cat)
        y_base = self.base_head(hidden)
        delta_y = self.residual_head(hidden)
        predicted_error = self.error_head(hidden) if self.use_error_head else torch.zeros_like(delta_y)
        alpha = self.gate(hidden, predicted_error)
        prediction = y_base + alpha * delta_y
        extras = {
            "y_base": y_base,
            "delta_y": delta_y,
            "predicted_error": predicted_error,
            "alpha": alpha,
        }
        if return_extras:
            return prediction.squeeze(-1), extras
        return prediction.squeeze(-1), {}


class AGRTMLP(TabModel):
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
        variant_name: str = "agr_tmlp",
    ):
        if feat_gate or pruning:
            raise NotImplementedError("agr_tmlp does not support sparse gating options yet")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.variant_name = variant_name
        self.model = _AGRTMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = variant_name
        self.device = torch.device(device)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.setdefault("residual_dropout_head", 0.1)
        model_config.setdefault("use_error_head", True)
        model_config.setdefault("gate_from_h_only", False)
        return model_config

    def _set_train_stage(self, epoch_idx: int, training_args: dict) -> None:
        freeze_epochs = training_args.get("freeze_backbone_epochs", 0)
        finetune_last = training_args.get("finetune_last_block", False)
        if freeze_epochs <= 0:
            self.model.set_backbone_trainable(True)
            return
        if epoch_idx < freeze_epochs:
            self.model.set_backbone_trainable(False)
        elif finetune_last:
            self.model.set_backbone_trainable(False)
            self.model.set_backbone_trainable(True, last_layer_only=True)
        else:
            self.model.set_backbone_trainable(True)

    @staticmethod
    def _corrcoef(predicted_error: np.ndarray, actual_error: np.ndarray) -> float:
        if len(predicted_error) <= 1:
            return 0.0
        pred_std = predicted_error.std()
        actual_std = actual_error.std()
        if pred_std < 1e-12 or actual_std < 1e-12:
            return 0.0
        return float(np.corrcoef(predicted_error, actual_error)[0, 1])

    def _evaluate_aux_stats(
        self,
        loader: ty.Tuple[DataLoader, int],
        placeholders,
    ) -> dict:
        self.model.eval()
        alpha_vals = []
        delta_vals = []
        pred_err_vals = []
        actual_err_vals = []
        for batch in loader:
            x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
            with torch.no_grad():
                predictions, extras = self.model(x_num, x_cat, return_extras=True)
            _ = predictions
            alpha_vals.append(extras["alpha"].detach().cpu().numpy())
            delta_vals.append(extras["delta_y"].abs().detach().cpu().numpy())
            pred_err_vals.append(extras["predicted_error"].detach().cpu().numpy())
            actual_err_vals.append((y - extras["y_base"].detach()).abs().cpu().numpy())
        self.model.train()
        alpha_all = np.concatenate(alpha_vals, axis=0).reshape(-1)
        delta_all = np.concatenate(delta_vals, axis=0).reshape(-1)
        pred_err_all = np.concatenate(pred_err_vals, axis=0).reshape(-1)
        actual_err_all = np.concatenate(actual_err_vals, axis=0).reshape(-1)
        return {
            "alpha_mean": float(alpha_all.mean()),
            "abs_delta_mean": float(delta_all.mean()),
            "predicted_error_mean": float(pred_err_all.mean()),
            "error_corr": self._corrcoef(pred_err_all, actual_err_all),
        }

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
            raise NotImplementedError("agr_tmlp variants currently support regression only")
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
        training_args.setdefault("lambda_err", 0.1)
        training_args.setdefault("lambda_gate", 1e-3)
        training_args.setdefault("lambda_res", 1e-4)
        training_args.setdefault("freeze_backbone_epochs", 0)
        training_args.setdefault("finetune_last_block", False)
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

        train_eval_loader = TabModel.prepare_tensor_loader(
            X_num=X_num,
            X_cat=X_cat,
            ys=ys,
            ids=ids,
            batch_size=training_args["batch_size"],
        )
        train_eval_placeholders = train_eval_loader[1]

        if eval_set is not None:
            dev_loader = TabModel.prepare_tensor_loader(
                X_num=eval_set[0][0], X_cat=eval_set[0][1], ys=eval_set[0][2], ids=eval_set[0][3],
                batch_size=training_args["batch_size"],
            )
            dev_placeholders = dev_loader[1]
            test_loader = None
            if len(eval_set) == 2:
                test_loader = TabModel.prepare_tensor_loader(
                    X_num=eval_set[1][0], X_cat=eval_set[1][1], ys=eval_set[1][2], ids=eval_set[1][3],
                    batch_size=training_args["batch_size"],
                )
        else:
            dev_loader, dev_placeholders, test_loader = None, None, None

        steps_per_epoch = len(train_loader)
        tot_step, tot_time = 0, 0.0
        for epoch_idx in range(training_args["max_epochs"]):
            self._set_train_stage(epoch_idx, training_args)
            self.model.train()
            tot_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
                start_time = time.time()
                predictions, extras = self.model(x_num, x_cat, return_extras=True)
                forward_time = time.time() - start_time

                reg_loss = F.huber_loss(predictions, y, delta=training_args["huber_delta"])
                target_err = (y - extras["y_base"].detach().squeeze(-1)).abs()
                predicted_err = extras["predicted_error"].squeeze(-1)
                err_loss = F.mse_loss(predicted_err, target_err)
                gate_loss = extras["alpha"].mean()
                res_loss = extras["delta_y"].pow(2).mean()

                loss = reg_loss
                if self.model.use_error_head:
                    loss = loss + training_args["lambda_err"] * err_loss
                loss = loss + training_args["lambda_gate"] * gate_loss
                loss = loss + training_args["lambda_res"] * res_loss

                start_time = time.time()
                loss.backward()
                backward_time = time.time() - start_time
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                tot_time += forward_time + backward_time
                tot_step += 1
                tot_loss += loss.detach().cpu().item()

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
                train_aux = self._evaluate_aux_stats(train_eval_loader[0], train_eval_placeholders)
                val_metrics = None
                val_aux = None
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
                    val_aux = self._evaluate_aux_stats(dev_loader[0], dev_placeholders)

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
                        f"[agr] train_alpha={train_aux['alpha_mean']:.6g}"
                        f" | train_abs_delta={train_aux['abs_delta_mean']:.6g}"
                        f" | train_pred_err={train_aux['predicted_error_mean']:.6g}"
                        f" | train_err_corr={train_aux['error_corr']:.6g}"
                        + (
                            f" | val_alpha={val_aux['alpha_mean']:.6g}"
                            f" | val_abs_delta={val_aux['abs_delta_mean']:.6g}"
                            f" | val_pred_err={val_aux['predicted_error_mean']:.6g}"
                            f" | val_err_corr={val_aux['error_corr']:.6g}"
                            if val_aux is not None else ""
                        )
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
            y_std=y_std,
            ids=ids,
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
