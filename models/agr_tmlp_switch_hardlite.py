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


class MicroCrossAdapter(nn.Module):
    def __init__(self, cross_rank: int, out_dim: int = 1) -> None:
        super().__init__()
        cross_rank = max(int(cross_rank), 4)
        hidden_dim = max(cross_rank, 16)
        self.scalar_proj = nn.Linear(1, cross_rank)
        self.cross_proj = nn.Linear(cross_rank, cross_rank)
        self.head = nn.Sequential(
            nn.LayerNorm(cross_rank),
            nn.Linear(cross_rank, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init()

    def _init(self) -> None:
        nn_init.normal_(self.scalar_proj.weight, mean=0.0, std=1e-3)
        nn_init.zeros_(self.scalar_proj.bias)
        nn_init.normal_(self.cross_proj.weight, mean=0.0, std=1e-3)
        nn_init.zeros_(self.cross_proj.bias)
        last = ty.cast(nn.Linear, self.head[-1])
        nn_init.normal_(last.weight, mean=0.0, std=1e-3)
        nn_init.zeros_(last.bias)

    def forward(self, topk_xw: Tensor) -> Tensor:
        # Tiny low-rank cross: project selected scalars, add one multiplicative interaction.
        tokens = self.scalar_proj(topk_xw.unsqueeze(-1))
        pooled = tokens.mean(dim=1)
        crossed = pooled * self.cross_proj(pooled)
        return self.head(pooled + crossed)


class _AGRTMLPSwitchHardLite(nn.Module):
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
        expert_hidden_ratio: float = 0.25,
        num_residual_experts: int = 2,
        topk_ratio: float = 0.15,
        topk_min: int = 4,
        topk_max: int = 8,
        cross_rank: int = 8,
        beta_init_bias: float = -2.0,
        expert_output_scale_init: float = 0.15,
        cross_output_scale_init: float = 0.1,
        mix_output_scale: float = 1.0,
        mode_temperature: float = 0.35,
        use_gumbel_mode_gate: bool = False,
        mode_sparsity_weight: float = 1e-4,
        mode_gate_bias_off: float = 2.0,
        mode_gate_bias_safe: float = 1.2,
        mode_gate_bias_expert: float = -0.8,
        mode_gate_bias_cross: float = -1.0,
        use_router: bool = True,
        router_temperature: float = 1.5,
        **backbone_config,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("agr_tmlp_switch_hardlite currently supports regression only")
        if d_numerical <= 0:
            raise NotImplementedError("agr_tmlp_switch_hardlite requires numerical features")
        if num_residual_experts != 2:
            raise NotImplementedError("agr_tmlp_switch_hardlite fixes num_residual_experts=2")
        if use_hidmix:
            raise NotImplementedError("agr_tmlp_switch_hardlite does not implement hidmix")

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
        self.use_router = use_router
        self.router_temperature = router_temperature
        self.topk_ratio = topk_ratio
        self.topk_min = int(topk_min)
        self.topk_max = int(topk_max)
        self.mix_output_scale = mix_output_scale
        self.mode_temperature = mode_temperature
        self.use_gumbel_mode_gate = use_gumbel_mode_gate
        self.mode_sparsity_weight = mode_sparsity_weight

        d_token = self.backbone.d_token
        gate_hidden = max(d_numerical // 2, 16)
        stat_dim = d_token + d_out + 3
        mode_dim = stat_dim + 7

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
        self.expert_head_1 = ResidualMLP(
            d_token,
            hidden_ratio=expert_hidden_ratio,
            out_dim=d_out,
            zero_last=False,
        )
        self.expert_head_2 = ResidualMLP(
            d_token,
            hidden_ratio=expert_hidden_ratio,
            out_dim=d_out,
            zero_last=False,
        )
        self.router = ResidualMLP(
            stat_dim,
            hidden_ratio=0.25,
            out_dim=2,
            zero_last=False,
        )
        self.beta_head = ResidualMLP(
            stat_dim,
            hidden_ratio=0.25,
            out_dim=1,
            last_bias_init=beta_init_bias,
            zero_last=False,
        )
        self.cross_adapter = MicroCrossAdapter(
            cross_rank=cross_rank,
            out_dim=d_out,
        )
        self.mode_gate = ResidualMLP(mode_dim, hidden_ratio=0.25, out_dim=4, zero_last=False)
        self.expert_output_scale = nn.Parameter(torch.tensor(expert_output_scale_init))
        self.cross_output_scale = nn.Parameter(torch.tensor(cross_output_scale_init))
        self._init_mode_gate(
            mode_gate_bias_off,
            mode_gate_bias_safe,
            mode_gate_bias_expert,
            mode_gate_bias_cross,
        )
        self._init_gate()

    def _init_gate(self) -> None:
        gate_last = ty.cast(nn.Linear, self.input_gate[-1])
        nn_init.zeros_(gate_last.weight)
        nn_init.zeros_(gate_last.bias)

    def _init_mode_gate(
        self,
        bias_off: float,
        bias_safe: float,
        bias_expert: float,
        bias_cross: float,
    ) -> None:
        last = ty.cast(nn.Linear, self.mode_gate.net[-1])
        nn_init.normal_(last.weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            last.bias.copy_(
                torch.tensor([bias_off, bias_safe, bias_expert, bias_cross], dtype=last.bias.dtype)
            )

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
            raise NotImplementedError("agr_tmlp_switch_hardlite requires numerical features")

        g = torch.sigmoid(self.input_gate(x_num))
        x_gated = x_num * g
        h_base, _ = self.backbone.encode_hidden(x_gated, x_cat)
        y_base = self.base_head(h_base)

        gate_mean, gate_max, gate_std = self._gate_stats(g)
        router_input = torch.cat([h_base, y_base, gate_mean, gate_max, gate_std], dim=1)

        delta_safe = self.safe_residual_head(h_base)
        e1 = self.expert_head_1(h_base)
        e2 = self.expert_head_2(h_base)

        if self.use_router:
            router_logits = self.router(router_input)
            alpha = torch.softmax(router_logits / max(self.router_temperature, 1e-6), dim=-1)
        else:
            alpha = torch.full(
                (h_base.shape[0], 2),
                0.5,
                dtype=h_base.dtype,
                device=h_base.device,
            )
        alpha_max = alpha.max(dim=1, keepdim=True).values
        alpha_entropy = -(alpha * torch.log(alpha.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        expert_gap = (e1 - e2).abs()

        delta_expert_raw = alpha[:, 0:1] * e1 + alpha[:, 1:2] * e2
        delta_expert = self.expert_output_scale * torch.tanh(delta_expert_raw)

        topk_x, topk_g = self._select_topk(x_num, g)
        topk_w = topk_g / (topk_g.sum(dim=1, keepdim=True) + 1e-6)
        topk_xw = topk_x * topk_w
        topk_gate_mean = topk_g.mean(dim=1, keepdim=True)
        topk_gate_max = topk_g.max(dim=1, keepdim=True).values
        topk_gate_std = topk_g.std(dim=1, unbiased=False, keepdim=True)
        topk_gate_concentration = topk_gate_max / (topk_gate_mean + 1e-6)
        delta_cross_raw = self.cross_adapter(topk_xw)
        delta_cross = self.cross_output_scale * torch.tanh(delta_cross_raw)

        beta = torch.sigmoid(self.beta_head(router_input))
        mode_input = torch.cat(
            [
                h_base,
                y_base,
                gate_mean,
                gate_max,
                gate_std,
                alpha_max,
                alpha_entropy,
                expert_gap,
                topk_gate_mean,
                topk_gate_max,
                topk_gate_std,
                topk_gate_concentration,
            ],
            dim=1,
        )
        mode_logits = self.mode_gate(mode_input)
        if self.use_gumbel_mode_gate and self.training:
            mode_probs = F.gumbel_softmax(
                mode_logits,
                tau=max(self.mode_temperature, 1e-6),
                hard=False,
                dim=-1,
            )
        else:
            mode_probs = torch.softmax(mode_logits / max(self.mode_temperature, 1e-6), dim=-1)
        w_off = mode_probs[:, 0:1]
        w_safe = mode_probs[:, 1:2]
        w_expert = mode_probs[:, 2:3]
        w_cross = mode_probs[:, 3:4]
        delta_mix_raw = w_safe * delta_safe + w_expert * delta_expert + w_cross * delta_cross
        delta_mix = self.mix_output_scale * torch.tanh(delta_mix_raw)

        # Harder mode switch: off mode suppresses correction via the softmax mass.
        y_hat = y_base + beta * delta_mix

        if not return_extras:
            return y_hat.squeeze(-1), {}

        return y_hat.squeeze(-1), {
            "y_base": y_base,
            "delta_safe": delta_safe,
            "delta_expert": delta_expert,
            "delta_cross": delta_cross,
            "delta_mix": delta_mix,
            "delta_mix_raw": delta_mix_raw,
            "beta": beta,
            "mode_probs": mode_probs,
            "mode_entropy": -(mode_probs * torch.log(mode_probs.clamp_min(1e-8))).sum(dim=1, keepdim=True),
            "w_off": w_off,
            "w_safe": w_safe,
            "w_expert": w_expert,
            "w_cross": w_cross,
            "topk_gate_mean": topk_gate_mean,
            "topk_gate_max": topk_gate_max,
            "topk_gate_std": topk_gate_std,
            "topk_gate_concentration": topk_gate_concentration,
            "alpha": alpha,
            "alpha_max": alpha_max,
            "alpha_entropy": alpha_entropy,
            "expert_gap": expert_gap,
            "gate_scores": g,
        }


class AGRTMLPSwitchHardLite(TabModel):
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
            raise NotImplementedError("agr_tmlp_switch_hardlite does not support sparse gating options yet")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _AGRTMLPSwitchHardLite(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = "agr_tmlp_switch_hardlite"
        self.device = torch.device(device)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.setdefault("residual_dropout_head", 0.0)
        model_config.setdefault("gate_reg_weight_scale", 0.05)
        model_config.setdefault("use_corr_reg", False)
        model_config.setdefault("use_featmix", False)
        model_config.setdefault("use_hidmix", False)
        model_config.setdefault("safe_hidden_ratio", 0.5)
        model_config.setdefault("expert_hidden_ratio", 0.25)
        model_config.setdefault("num_residual_experts", 2)
        model_config.setdefault("topk_ratio", 0.15)
        model_config.setdefault("topk_min", 4)
        model_config.setdefault("topk_max", 8)
        model_config.setdefault("cross_rank", 8)
        model_config.setdefault("beta_init_bias", -2.0)
        model_config.setdefault("expert_output_scale_init", 0.15)
        model_config.setdefault("cross_output_scale_init", 0.1)
        model_config.setdefault("mix_output_scale", 1.0)
        model_config.setdefault("mode_temperature", 0.35)
        model_config.setdefault("use_gumbel_mode_gate", False)
        model_config.setdefault("mode_sparsity_weight", 1e-4)
        model_config.setdefault("mode_gate_bias_off", 2.0)
        model_config.setdefault("mode_gate_bias_safe", 1.2)
        model_config.setdefault("mode_gate_bias_expert", -0.8)
        model_config.setdefault("mode_gate_bias_cross", -1.0)
        model_config.setdefault("use_router", True)
        model_config.setdefault("router_temperature", 1.5)
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
            raise NotImplementedError("agr_tmlp_switch_hardlite currently supports regression only")
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
        training_args.setdefault("lambda_gate", 1e-3)
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
            safe_vals = []
            expert_vals = []
            cross_vals = []
            off_vals = []
            safe_mode_vals = []
            expert_mode_vals = []
            cross_mode_vals = []
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
                gate_reg = extras["gate_scores"].mean()
                loss = loss + (
                    training_args["lambda_gate"]
                    * self.model.gate_reg_weight_scale
                    * gate_reg
                )
                if self.model.use_corr_reg:
                    corr_term = (
                        extras["delta_safe"].pow(2).mean()
                        + extras["delta_expert"].pow(2).mean()
                        + extras["delta_cross"].pow(2).mean()
                    )
                    loss = loss + training_args["lambda_corr"] * corr_term
                if self.model.mode_sparsity_weight > 0:
                    loss = loss + self.model.mode_sparsity_weight * extras["mode_entropy"].mean()

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
                safe_vals.append(float(extras["delta_safe"].detach().abs().mean().cpu().item()))
                expert_vals.append(float(extras["delta_expert"].detach().abs().mean().cpu().item()))
                cross_vals.append(float(extras["delta_cross"].detach().abs().mean().cpu().item()))
                off_vals.append(float(extras["w_off"].detach().mean().cpu().item()))
                safe_mode_vals.append(float(extras["w_safe"].detach().mean().cpu().item()))
                expert_mode_vals.append(float(extras["w_expert"].detach().mean().cpu().item()))
                cross_mode_vals.append(float(extras["w_cross"].detach().mean().cpu().item()))

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
                        f"[switch_hardlite] beta_mean={np.mean(beta_vals):.6g}"
                        f" | abs_delta_safe={np.mean(safe_vals):.6g}"
                        f" | abs_delta_expert={np.mean(expert_vals):.6g}"
                        f" | abs_delta_cross={np.mean(cross_vals):.6g}"
                        f" | w_off={np.mean(off_vals):.6g}"
                        f" | w_safe={np.mean(safe_mode_vals):.6g}"
                        f" | w_expert={np.mean(expert_mode_vals):.6g}"
                        f" | w_cross={np.mean(cross_mode_vals):.6g}"
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
