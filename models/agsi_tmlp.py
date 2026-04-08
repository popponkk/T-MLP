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


class AGSISpaBlock(nn.Module):
    def __init__(self, token_dim: int, n_heads: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if token_dim % n_heads != 0:
            raise ValueError(f"token_dim={token_dim} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = token_dim // n_heads
        self.norm1 = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, max(token_dim * 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(token_dim * 2, 16), token_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, token_dim = x.shape
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, n_tokens, _ = x.shape
        return x.transpose(1, 2).reshape(batch_size, n_tokens, self.n_heads * self.head_dim)

    def forward(self, tokens: Tensor, importance: Tensor) -> Tensor:
        # Selective interaction: more important selected features bias attention keys.
        residual = tokens
        x = self.norm1(tokens)
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + importance[:, None, None, :]
        attn = torch.softmax(attn_scores, dim=-1)
        attn_out = self._merge_heads(torch.matmul(attn, v))
        x = residual + self.dropout(self.out_proj(attn_out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class _AGSITMLP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        token_dim: int = 16,
        spa_num_heads: int = 2,
        topk_ratio: float = 0.25,
        topk_min: int = 8,
        topk_max: int = 16,
        top_k: ty.Optional[int] = None,
        beta_init_bias: float = -2.0,
        use_gate_stats_for_beta: bool = True,
        use_featmix: bool = False,
        use_hidmix: bool = False,
        use_gate_reg: bool = False,
        use_corr_reg: bool = False,
        dropout: float = 0.1,
        **backbone_config,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("AGSI-TMLP currently supports regression only")
        if d_numerical <= 0:
            raise NotImplementedError("AGSI-TMLP requires at least one numerical feature")
        if use_hidmix:
            raise NotImplementedError("AGSI-TMLP does not implement Hid-Mix in the first version")

        self.backbone = TMLPBackbone(
            d_numerical=d_numerical,
            categories=categories,
            **backbone_config,
        )
        self.n_num_features = d_numerical
        self.token_dim = token_dim
        self.top_k = top_k or min(d_numerical, max(topk_min, min(topk_max, round(topk_ratio * d_numerical))))
        self.use_gate_stats_for_beta = use_gate_stats_for_beta
        self.use_featmix = use_featmix
        self.use_gate_reg = use_gate_reg
        self.use_corr_reg = use_corr_reg

        d_token = self.backbone.d_token
        self.input_gate = nn.Linear(d_numerical, d_numerical)
        self.value_proj = nn.Parameter(torch.empty(d_numerical, token_dim))
        self.feature_embed = nn.Parameter(torch.empty(d_numerical, token_dim))
        self.spa_block = AGSISpaBlock(token_dim, n_heads=spa_num_heads, dropout=dropout)

        # Baseline path: original TMLP backbone hidden state and regression head.
        self.base_head = nn.Linear(d_token, d_out)

        # Selective interaction branch: top-k scalar features -> tokens -> SPA -> correction.
        delta_hidden = max((d_token + token_dim) // 4, 32)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(d_token + token_dim),
            nn.Linear(d_token + token_dim, delta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(delta_hidden, d_out),
        )

        beta_in = d_token + d_out + (3 if use_gate_stats_for_beta else 0)
        beta_hidden = max(d_token // 4, 16)
        self.beta_head = nn.Sequential(
            nn.LayerNorm(beta_in),
            nn.Linear(beta_in, beta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(beta_hidden, 1),
        )
        self._init_lightweight_params(beta_init_bias)

    def _init_lightweight_params(self, beta_init_bias: float) -> None:
        nn.init.kaiming_uniform_(self.value_proj, a=math.sqrt(5))
        nn.init.normal_(self.feature_embed, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_gate.weight)
        nn.init.zeros_(self.input_gate.bias)
        delta_last = ty.cast(nn.Linear, self.delta_head[-1])
        nn.init.zeros_(delta_last.weight)
        nn.init.zeros_(delta_last.bias)
        beta_last = ty.cast(nn.Linear, self.beta_head[-1])
        nn.init.zeros_(beta_last.weight)
        nn.init.constant_(beta_last.bias, beta_init_bias)

    def _select_features(self, x_num: Tensor, gate_scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        k = min(self.top_k, x_num.shape[1])
        topk_g, topk_idx = torch.topk(gate_scores, k=k, dim=1)
        topk_x = torch.gather(x_num, 1, topk_idx)
        topk_w = topk_g / (topk_g.sum(dim=1, keepdim=True) + 1e-6)
        return topk_x, topk_g, topk_w, topk_idx

    def _tokenize_selected(self, topk_x: Tensor, topk_w: Tensor, topk_idx: Tensor) -> Tensor:
        value_proj = self.value_proj[topk_idx]
        feature_embed = self.feature_embed[topk_idx]
        tokens = topk_x.unsqueeze(-1) * value_proj + feature_embed
        return tokens * topk_w.unsqueeze(-1)

    @staticmethod
    def _gate_stats(gate_scores: Tensor) -> Tensor:
        mean = gate_scores.mean(dim=1, keepdim=True)
        max_value = gate_scores.max(dim=1, keepdim=True).values
        std = gate_scores.std(dim=1, unbiased=False, keepdim=True)
        return torch.cat([mean, max_value, std], dim=1)

    def forward(self, x_num, x_cat, return_extras: bool = False):
        if x_num is None:
            raise NotImplementedError("AGSI-TMLP requires numerical features")
        # Baseline: gated numerical inputs go through the original TMLP-style backbone.
        gate_scores = torch.sigmoid(self.input_gate(x_num))
        x_gated = x_num * gate_scores
        h_base, _ = self.backbone.encode_hidden(x_gated, x_cat)
        y_base = self.base_head(h_base)

        # Selective interaction: only top-k gate-selected numerical features interact.
        topk_x, topk_g, topk_w, topk_idx = self._select_features(x_num, gate_scores)
        tokens = self._tokenize_selected(topk_x, topk_w, topk_idx)
        z_tokens = self.spa_block(tokens, topk_g)
        z_int = z_tokens.mean(dim=1)
        delta_y_int = self.delta_head(torch.cat([h_base, z_int], dim=1))

        # Adaptive fusion: decide per sample whether to apply the interaction correction.
        beta_parts = [h_base, y_base]
        if self.use_gate_stats_for_beta:
            beta_parts.append(self._gate_stats(gate_scores))
        beta = torch.sigmoid(self.beta_head(torch.cat(beta_parts, dim=1)))
        prediction = y_base + beta * delta_y_int

        if not return_extras:
            return prediction.squeeze(-1), {}

        return prediction.squeeze(-1), {
            "y_base": y_base,
            "delta_y_int": delta_y_int,
            "beta": beta,
            "gate_scores": gate_scores,
            "selected_count": float(topk_x.shape[1]),
        }


class AGSITMLP(TabModel):
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
            raise NotImplementedError("agsi_tmlp does not support sparse gating options yet")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _AGSITMLP(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "agsi_tmlp"
        self.device = torch.device(device)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.setdefault("token_dim", 16)
        model_config.setdefault("spa_num_heads", 2)
        model_config.setdefault("topk_ratio", 0.25)
        model_config.setdefault("topk_min", 8)
        model_config.setdefault("topk_max", 16)
        model_config.setdefault("top_k", None)
        model_config.setdefault("beta_init_bias", -2.0)
        model_config.setdefault("use_gate_stats_for_beta", True)
        model_config.setdefault("use_featmix", False)
        model_config.setdefault("use_hidmix", False)
        model_config.setdefault("use_gate_reg", False)
        model_config.setdefault("use_corr_reg", False)
        model_config.setdefault("dropout", 0.1)
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
            raise NotImplementedError("agsi_tmlp currently supports regression only")
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
            select_vals = []
            gate_vals = []
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
                    loss = loss + training_args["lambda_corr"] * extras["delta_y_int"].pow(2).mean()

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
                delta_vals.append(float(extras["delta_y_int"].detach().abs().mean().cpu().item()))
                select_vals.append(float(extras["selected_count"]))
                gate_vals.append(extras["gate_scores"].detach().mean(dim=0).cpu().numpy())

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
                        f"[agsi] beta_mean={np.mean(beta_vals):.6g}"
                        f" | abs_delta_y_int={np.mean(delta_vals):.6g}"
                        f" | selected_features={np.mean(select_vals):.6g}"
                        f" | gate_importance={np.mean(np.stack(gate_vals, axis=0), axis=0).tolist()}"
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
