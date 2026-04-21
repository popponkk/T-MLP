import time
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from torch.utils.data import DataLoader

from models.abstract import TabModel, check_dir
from models.excel_former import Tokenizer, MultiheadAttention, attenuated_kaiming_uniform_
from utils.deep import tanglu


class LiteResidualHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int = 1,
        last_bias_init: float = 0.0,
        last_weight_std: float = 1e-4,
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


class _ExcelBackboneForCGR(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        prenormalization: bool,
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        d_out: int,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)
        n_tokens = d_numerical + len(categories) if categories is not None else d_numerical
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        self.n_categories = 0 if categories is None else len(categories)

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(n_tokens, int(n_tokens * kv_compression), bias=False)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, init_scale=init_scale
                    ),
                    'linear0': nn.Linear(d_token, d_token * 2),
                    'norm1': make_normalization(),
                }
            )
            attenuated_kaiming_uniform_(layer['linear0'].weight, scale=init_scale)
            nn_init.zeros_(layer['linear0'].bias)

            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = tanglu
        self.last_activation = nn.PReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.base_head = nn.Linear(d_token, d_out)
        attenuated_kaiming_uniform_(self.base_head.weight)
        self.last_fc = nn.Linear(n_tokens, 1)
        attenuated_kaiming_uniform_(self.last_fc.weight)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def encode_hidden(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> tuple[Tensor, Tensor]:
        assert x_cat is None
        x = self.tokenizer(x_num)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        h = self.last_fc(x.transpose(1, 2))[:, :, 0]
        if self.last_normalization is not None:
            h = self.last_normalization(h)
        h = self.last_activation(h)
        return h, x

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> tuple[Tensor, Tensor]:
        h, token_states = self.encode_hidden(x_num, x_cat)
        y_base = self.base_head(h)
        return y_base.squeeze(-1), h


class _ExcelCGRLite(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_out: int,
        token_bias: bool = True,
        n_layers: int = 3,
        d_token: int = 256,
        n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        prenormalization: bool = True,
        kv_compression: ty.Optional[float] = None,
        kv_compression_sharing: ty.Optional[str] = None,
        init_scale: float = 0.1,
        use_cgr_refinement: bool = True,
        safe_hidden_dim: int = 32,
        spec_hidden_dim: int = 32,
        spec_scale: float = 0.1,
        gamma_init_bias: float = -2.5,
    ) -> None:
        super().__init__()
        if d_out != 1:
            raise NotImplementedError("excel_cgr_lite currently supports scalar regression only")
        if categories is not None:
            raise NotImplementedError("excel_cgr_lite follows the current ExcelFormer implementation and supports numerical features only")
        if not use_cgr_refinement:
            raise ValueError("excel_cgr_lite requires use_cgr_refinement=True")

        self.use_featmix = False
        self.use_corr_reg = False
        self.use_hidmix = False
        self.spec_scale = float(spec_scale)

        self.backbone = _ExcelBackboneForCGR(
            d_numerical=d_numerical,
            categories=categories,
            token_bias=token_bias,
            n_layers=n_layers,
            d_token=d_token,
            n_heads=n_heads,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
            prenormalization=prenormalization,
            kv_compression=kv_compression,
            kv_compression_sharing=kv_compression_sharing,
            d_out=d_out,
            init_scale=init_scale,
        )
        self.safe_head = LiteResidualHead(d_token, safe_hidden_dim, out_dim=1, last_weight_std=1e-4)
        self.spec_head = LiteResidualHead(d_token + 1, spec_hidden_dim, out_dim=1, last_weight_std=1e-4)
        self.conf_gate = LiteResidualHead(d_token + 3, spec_hidden_dim, out_dim=1, last_bias_init=gamma_init_bias, last_weight_std=1e-4)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor], return_extras: bool = False):
        y_base, h = self.backbone(x_num, x_cat)
        y_base_unsq = y_base.unsqueeze(-1)
        delta_safe = self.safe_head(h)
        spec_input = torch.cat([h, y_base_unsq], dim=1)
        raw_delta_spec = self.spec_head(spec_input)
        delta_spec = self.spec_scale * torch.tanh(raw_delta_spec)
        conf_input = torch.cat([h, y_base_unsq, delta_safe.abs(), delta_spec.abs()], dim=1)
        gamma = torch.sigmoid(self.conf_gate(conf_input))
        y_hat = y_base_unsq + delta_safe + gamma * delta_spec

        if not return_extras:
            return y_hat.squeeze(-1), {}
        return y_hat.squeeze(-1), {
            'h': h,
            'y_base': y_base_unsq,
            'delta_safe': delta_safe,
            'delta_spec': delta_spec,
            'gamma': gamma,
        }


class ExcelCGRLite(TabModel):
    def __init__(
        self,
        model_config: dict,
        n_num_features: int,
        categories: ty.Optional[ty.List[int]],
        n_labels: int,
        device: ty.Union[str, torch.device] = 'cuda',
        feat_gate: ty.Optional[str] = None,
        pruning: ty.Optional[str] = None,
        dataset=None,
    ):
        if feat_gate or pruning:
            raise NotImplementedError("excel_cgr_lite does not support sparse gating options")
        super().__init__()
        model_config = self.preproc_config(model_config)
        self.model = _ExcelCGRLite(
            d_numerical=n_num_features,
            categories=categories,
            d_out=n_labels,
            **model_config,
        ).to(device)
        self.base_name = 'excel_cgr_lite'
        self.device = torch.device(device)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop('model_name', None)
        model_config.pop('base_model', None)
        model_config.setdefault('token_bias', True)
        model_config.setdefault('kv_compression', None)
        model_config.setdefault('kv_compression_sharing', None)
        if model_config['d_token'] % model_config['n_heads'] != 0:
            model_config['d_token'] = model_config['d_token'] // model_config['n_heads'] * model_config['n_heads']
        model_config.setdefault('use_cgr_refinement', True)
        model_config.setdefault('safe_hidden_dim', 32)
        model_config.setdefault('spec_hidden_dim', 32)
        model_config.setdefault('spec_scale', 0.1)
        model_config.setdefault('gamma_init_bias', -2.5)
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
        def train_step(model, x_num, x_cat, y):
            start_time = time.time()
            logits, _ = model(x_num, x_cat, return_extras=False)
            used_time = time.time() - start_time
            return logits, used_time

        self.dnn_fit(
            dnn_fit_func=train_step,
            train_loader=train_loader,
            X_num=X_num, X_cat=X_cat, ys=ys, y_std=y_std, ids=ids,
            eval_set=eval_set, patience=patience, task=task,
            training_args=training_args,
            meta_args=meta_args,
        )

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
            start_time = time.time()
            logits, _ = model(x_num, x_cat, return_extras=False)
            used_time = time.time() - start_time
            return logits, used_time

        return self.dnn_predict(
            dnn_predict_func=inference_step,
            dev_loader=dev_loader,
            X_num=X_num, X_cat=X_cat, ys=ys, y_std=y_std, ids=ids, task=task,
            return_probs=return_probs, return_metric=return_metric, return_loss=return_loss,
            meta_args=meta_args,
        )

    def save(self, output_dir):
        check_dir(output_dir)
        self.save_pt_model(output_dir)
        self.save_history(output_dir)
        self.save_config(output_dir)
