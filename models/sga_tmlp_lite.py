import math
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


class SelectiveAttentionBlock(nn.Module):
    def __init__(self, token_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = max(token_dim * 2, 16)
        self.norm1 = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, importance: Tensor) -> Tensor:
        residual = tokens
        x = self.norm1(tokens)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        key_bias = importance.unsqueeze(1)
        attn = torch.softmax(attn_scores + key_bias, dim=-1)
        attn_out = torch.matmul(attn, v)
        x = residual + self.dropout(self.out_proj(attn_out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class _SGATMLPLite(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        top_k: int = 8,
        token_dim: int = 16,
        num_spa_blocks: int = 1,
        dropout: float = 0.1,
        lambda_beta: float = 1e-3,
        lambda_corr: float = 1e-4,
        enable_featmix: bool = True,
        **backbone_config,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("SGA-TMLP-Lite currently supports regression only")

        self.backbone = TMLPBackbone(
            d_numerical=d_numerical,
            categories=categories,
            **backbone_config,
        )
        self.n_num_features = d_numerical
        self.top_k = top_k
        self.token_dim = token_dim
        self.num_spa_blocks = num_spa_blocks
        self.lambda_beta = lambda_beta
        self.lambda_corr = lambda_corr
        self.enable_featmix = enable_featmix

        d_token = self.backbone.d_token
        selector_hidden = max(d_token // 4, 16)
        fusion_hidden = max(d_token // 4, 16)
        correction_hidden = max(token_dim * 2, 16)

        self.base_head = nn.Linear(d_token, d_out)
        self.feature_selector = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, selector_hidden),
            nn.GELU(),
            nn.Linear(selector_hidden, 1),
        )
        self.token_projector = nn.Linear(d_token, token_dim)
        self.spa_blocks = nn.ModuleList(
            [SelectiveAttentionBlock(token_dim, dropout=dropout) for _ in range(num_spa_blocks)]
        )
        self.correction_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, correction_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(correction_hidden, 1),
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(d_token + token_dim),
            nn.Linear(d_token + token_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )
        self._init_lightweight_heads()

    def _init_lightweight_heads(self) -> None:
        for module in self.correction_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(module.bias)
        gate_last = ty.cast(nn.Linear, self.fusion_gate[-1])
        nn.init.zeros_(gate_last.weight)
        nn.init.constant_(gate_last.bias, -4.0)

    def _select_topk(self, feature_tokens: Tensor, scores: Tensor) -> tuple[Tensor, Tensor]:
        k = min(self.top_k, feature_tokens.shape[1])
        top_scores, top_indices = torch.topk(scores, k=k, dim=1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, feature_tokens.shape[-1])
        selected_tokens = torch.gather(feature_tokens, 1, gather_index)
        return selected_tokens, top_scores

    def forward(self, x_num, x_cat, return_extras: bool = False):
        cls_hidden, token_states = self.backbone.encode_hidden(x_num, x_cat)
        feature_tokens = token_states[:, 1: 1 + self.n_num_features]
        if feature_tokens.shape[1] == 0:
            feature_tokens = token_states[:, 1:]
        if feature_tokens.shape[1] == 0:
            feature_tokens = cls_hidden.unsqueeze(1)

        feature_scores = self.feature_selector(feature_tokens).squeeze(-1)
        selected_tokens, selected_scores = self._select_topk(feature_tokens, feature_scores)
        interaction_tokens = self.token_projector(selected_tokens)
        for spa_block in self.spa_blocks:
            interaction_tokens = spa_block(interaction_tokens, selected_scores)
        interaction_summary = interaction_tokens.mean(dim=1)

        y_base = self.base_head(cls_hidden)
        delta_y_spa = self.correction_head(interaction_summary)
        beta_input = torch.cat([cls_hidden, interaction_summary], dim=1)
        beta = torch.sigmoid(self.fusion_gate(beta_input))
        prediction = y_base + beta * delta_y_spa

        if not return_extras:
            return prediction.squeeze(-1), {}

        importance_dist = torch.softmax(feature_scores, dim=1).mean(dim=0)
        return prediction.squeeze(-1), {
            "y_base": y_base,
            "delta_y_spa": delta_y_spa,
            "beta": beta,
            "selected_count": float(selected_tokens.shape[1]),
            "importance_dist": importance_dist.detach().cpu().tolist(),
        }


class SGATMLPLite(TabModel):
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
        variant_name: str = "sga_tmlp_lite",
    ):
        if feat_gate or pruning:
            raise NotImplementedError("sga_tmlp_lite does not support sparse gating options yet")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.variant_name = variant_name
        self.model = _SGATMLPLite(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = variant_name
        self.device = torch.device(device)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.setdefault("top_k", 8)
        model_config.setdefault("token_dim", 16)
        model_config.setdefault("num_spa_blocks", 1)
        model_config.setdefault("dropout", 0.1)
        model_config.setdefault("lambda_beta", 1e-3)
        model_config.setdefault("lambda_corr", 1e-4)
        model_config.setdefault("enable_featmix", True)
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
            raise NotImplementedError("sga_tmlp_lite currently supports regression only")
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
            corr_vals = []
            select_vals = []
            importance_snapshots = []
            for batch in train_loader:
                optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
                mixed_num, mixed_y = self._apply_featmix(
                    x_num,
                    y,
                    enabled=self.model.enable_featmix,
                    alpha=training_args["featmix_alpha"],
                )
                start_time = time.time()
                predictions, extras = self.model(mixed_num, x_cat, return_extras=True)
                forward_time = time.time() - start_time

                reg_loss = F.huber_loss(predictions, mixed_y, delta=training_args["huber_delta"])
                beta_loss = extras["beta"].mean()
                corr_loss = extras["delta_y_spa"].pow(2).mean()
                loss = reg_loss + self.model.lambda_beta * beta_loss + self.model.lambda_corr * corr_loss

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
                corr_vals.append(float(extras["delta_y_spa"].detach().abs().mean().cpu().item()))
                select_vals.append(float(extras["selected_count"]))
                importance_snapshots.append(extras["importance_dist"])

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
                        f"[sga] beta_mean={np.mean(beta_vals):.6g}"
                        f" | abs_delta_y_spa={np.mean(corr_vals):.6g}"
                        f" | selected_features={np.mean(select_vals):.6g}"
                        f" | avg_gate_importance={np.mean(np.stack(importance_snapshots, axis=0), axis=0).tolist()}"
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
