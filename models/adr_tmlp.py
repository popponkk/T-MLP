import time
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .abstract import TabModel, check_dir
from .tmlp_variant_blocks import TMLPBackbone


class TokenAttentiveReadout(nn.Module):
    def __init__(self, d_token: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = max(d_token // 4, 16)
        self.score = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature_tokens: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.score(feature_tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        summary = torch.einsum("bn,bnd->bd", weights, feature_tokens)
        return summary, weights


class _ADRTMLP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        readout_dropout: float = 0.1,
        **backbone_config,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("ADR-TMLP currently supports regression only")

        self.backbone = TMLPBackbone(
            d_numerical=d_numerical,
            categories=categories,
            **backbone_config,
        )
        d_token = self.backbone.d_token
        gate_hidden = max(d_token // 4, 16)
        token_hidden = max(d_token // 2, 16)

        self.cls_head = nn.Linear(d_token, d_out)
        self.readout = TokenAttentiveReadout(d_token, dropout=readout_dropout)
        self.token_head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, token_hidden),
            nn.GELU(),
            nn.Dropout(readout_dropout),
            nn.Linear(token_hidden, d_out),
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(d_token * 3),
            nn.Linear(d_token * 3, gate_hidden),
            nn.GELU(),
            nn.Dropout(readout_dropout),
            nn.Linear(gate_hidden, 1),
        )
        self._init_gate()

    def _init_gate(self) -> None:
        final = ty.cast(nn.Linear, self.fusion_gate[-1])
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, -2.0)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def forward(self, x_num, x_cat, return_extras: bool = False):
        cls_hidden, token_states = self.backbone.encode_hidden(x_num, x_cat)
        feature_tokens = token_states[:, 1:]
        if feature_tokens.shape[1] == 0:
            feature_tokens = cls_hidden.unsqueeze(1)

        token_summary, token_weights = self.readout(feature_tokens)
        y_cls = self.cls_head(cls_hidden)
        y_token = self.token_head(token_summary)

        gate_input = torch.cat(
            [cls_hidden, token_summary, (cls_hidden - token_summary).abs()],
            dim=1,
        )
        fusion = torch.sigmoid(self.fusion_gate(gate_input))
        prediction = y_cls + fusion * (y_token - y_cls)

        if not return_extras:
            return prediction.squeeze(-1), {}

        token_entropy = -(token_weights * token_weights.clamp_min(1e-8).log()).sum(dim=1).mean()
        return prediction.squeeze(-1), {
            "y_cls": y_cls,
            "y_token": y_token,
            "fusion": fusion,
            "token_weights": token_weights,
            "token_entropy": token_entropy,
        }


class ADRTMLP(TabModel):
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
            raise NotImplementedError("adr_tmlp does not support sparse gating options yet")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _ADRTMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "adr_tmlp"
        self.device = torch.device(device)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.setdefault("readout_dropout", 0.1)
        return model_config

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
            raise NotImplementedError("adr_tmlp currently supports regression only")
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
            fusion_vals = []
            entropy_vals = []
            for batch in train_loader:
                optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
                start_time = time.time()
                predictions, extras = self.model(x_num, x_cat, return_extras=True)
                forward_time = time.time() - start_time

                loss = F.huber_loss(predictions, y, delta=training_args["huber_delta"])

                start_time = time.time()
                loss.backward()
                backward_time = time.time() - start_time
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                tot_time += forward_time + backward_time
                tot_step += 1
                tot_loss += loss.detach().cpu().item()
                fusion_vals.append(float(extras["fusion"].detach().mean().cpu().item()))
                entropy_vals.append(float(extras["token_entropy"].detach().cpu().item()))

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
                        f"[adr] fusion_mean={np.mean(fusion_vals):.6g}"
                        f" | token_entropy={np.mean(entropy_vals):.6g}"
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
