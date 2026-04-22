import pickle
import time
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from models.abstract import TabModel, check_dir


def _merge_features(x_num, x_cat):
    if x_num is None and x_cat is None:
        raise ValueError("At least one of X_num or X_cat must be provided.")
    if x_num is None:
        return x_cat
    if x_cat is None:
        return x_num
    return np.concatenate([x_num, x_cat], axis=1)


class RefinementMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        refine_hidden_dim: int = 128,
        refine_num_layers: int = 2,
        safe_hidden_dim: int = 32,
        spec_hidden_dim: int = 32,
        spec_scale: float = 0.1,
        gamma_init_bias: float = -2.5,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(refine_hidden_dim), 16)
        num_layers = max(int(refine_num_layers), 1)
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.refine_trunk = nn.Sequential(*layers)
        self.safe_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(int(safe_hidden_dim), 8)),
            nn.ReLU(),
            nn.Linear(max(int(safe_hidden_dim), 8), 1),
        )
        self.spec_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 1),
            nn.Linear(hidden_dim + 1, max(int(spec_hidden_dim), 8)),
            nn.ReLU(),
            nn.Linear(max(int(spec_hidden_dim), 8), 1),
        )
        self.conf_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3),
            nn.Linear(hidden_dim + 3, max(int(spec_hidden_dim), 8)),
            nn.ReLU(),
            nn.Linear(max(int(spec_hidden_dim), 8), 1),
        )
        self.spec_scale = float(spec_scale)
        self._init(gamma_init_bias)

    def _init(self, gamma_init_bias: float) -> None:
        for module in self.refine_trunk:
            if isinstance(module, nn.Linear):
                nn_init.kaiming_uniform_(module.weight, a=5 ** 0.5)
                nn_init.zeros_(module.bias)
        for head in [self.safe_head, self.spec_head, self.conf_gate]:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn_init.normal_(module.weight, mean=0.0, std=1e-3)
                    nn_init.zeros_(module.bias)
        nn_init.normal_(self.safe_head[-1].weight, mean=0.0, std=1e-4)
        nn_init.normal_(self.spec_head[-1].weight, mean=0.0, std=1e-4)
        nn_init.normal_(self.conf_gate[-1].weight, mean=0.0, std=1e-4)
        nn_init.constant_(self.conf_gate[-1].bias, gamma_init_bias)

    def forward(self, x_features: Tensor, y_tree: Tensor) -> tuple[Tensor, dict]:
        ref_input = torch.cat([x_features, y_tree], dim=1)
        h = self.refine_trunk(ref_input)
        delta_safe = self.safe_head(h)
        spec_input = torch.cat([h, y_tree], dim=1)
        raw_delta_spec = self.spec_head(spec_input)
        delta_spec = self.spec_scale * torch.tanh(raw_delta_spec)
        conf_input = torch.cat([h, y_tree, delta_safe.abs(), delta_spec.abs()], dim=1)
        gamma = torch.sigmoid(self.conf_gate(conf_input))
        y_hat = y_tree + delta_safe + gamma * delta_spec
        return y_hat, {
            'h': h,
            'y_tree': y_tree,
            'delta_safe': delta_safe,
            'delta_spec': delta_spec,
            'gamma': gamma,
        }


class LGBMCGRHybrid(TabModel):
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
            raise NotImplementedError("lgbm_cgr_hybrid does not support sparse gating options")
        super().__init__()
        model_config = self.preproc_config(model_config)
        refine_input_dim = n_num_features + (0 if categories is None else len(categories))
        self.model = RefinementMLP(
            input_dim=refine_input_dim,
            refine_hidden_dim=model_config['refine_hidden_dim'],
            refine_num_layers=model_config['refine_num_layers'],
            safe_hidden_dim=model_config['safe_hidden_dim'],
            spec_hidden_dim=model_config['spec_hidden_dim'],
            spec_scale=model_config['spec_scale'],
            gamma_init_bias=model_config['gamma_init_bias'],
        ).to(device)
        self.base_name = 'lgbm_cgr_hybrid'
        self.device = torch.device(device)
        self.lgbm_model = None
        self.lgbm_params = self.saved_model_config.get('lgbm_params', {})
        self.use_lgbm_base = self.saved_model_config.get('use_lgbm_base', True)
        self.lgbm_cache_model = self.saved_model_config.get('lgbm_cache_model', True)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop('model_name', None)
        model_config.pop('use_lgbm_base', None)
        model_config.pop('lgbm_cache_model', None)
        model_config.setdefault('lgbm_params', {
            'n_estimators': 512,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        })
        model_config.setdefault('refine_hidden_dim', 128)
        model_config.setdefault('refine_num_layers', 2)
        model_config.setdefault('safe_hidden_dim', 32)
        model_config.setdefault('spec_hidden_dim', 32)
        model_config.setdefault('spec_scale', 0.1)
        model_config.setdefault('gamma_init_bias', -2.5)
        return model_config

    def _artifact_dir(self, save_path: str) -> Path:
        dataset_name = Path(save_path).name
        return Path('artifacts') / 'lgbm_cgr_hybrid' / dataset_name

    def _lgbm_model_file(self, save_path: str) -> Path:
        return self._artifact_dir(save_path) / 'lgbm_model.pkl'

    def _to_numpy(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _fit_or_load_lgbm(self, X_train, y_train, X_val, y_val, save_path: str, training_args: dict):
        model_file = self._lgbm_model_file(save_path)
        if self.lgbm_cache_model and model_file.exists():
            with open(model_file, 'rb') as f:
                self.lgbm_model = pickle.load(f)
            print(f"[lgbm_cgr_hybrid] loaded LightGBM model: {model_file}")
            return

        import lightgbm as lgb

        self.lgbm_model = lgb.LGBMRegressor(**self.lgbm_params)
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            if training_args.get('lgbm_early_stopping_rounds') is not None:
                fit_kwargs['callbacks'] = [
                    lgb.early_stopping(training_args['lgbm_early_stopping_rounds'], verbose=False),
                    lgb.log_evaluation(0),
                ]
        self.lgbm_model.fit(X_train, y_train, **fit_kwargs)

        if self.lgbm_cache_model:
            model_file.parent.mkdir(parents=True, exist_ok=True)
            with open(model_file, 'wb') as f:
                pickle.dump(self.lgbm_model, f)
            print(f"[lgbm_cgr_hybrid] saved LightGBM model: {model_file}")

    def _predict_tree(self, X):
        return self.lgbm_model.predict(X).reshape(-1, 1).astype('float32')

    def _prepare_refine_tensor(self, X_num, X_cat):
        if X_num is None and X_cat is None:
            raise ValueError("At least one of X_num or X_cat must be provided")
        tensors = []
        if X_num is not None:
            tensors.append(X_num.float())
        if X_cat is not None:
            tensors.append(X_cat.float())
        return torch.cat(tensors, dim=1)

    def fit(
        self,
        X_num=None,
        X_cat=None,
        ys=None,
        ids=None,
        y_std=None,
        eval_set=None,
        patience: int = 0,
        task: str = None,
        training_args: dict = None,
        meta_args: ty.Optional[dict] = None,
    ):
        if task != 'regression':
            raise NotImplementedError("lgbm_cgr_hybrid currently supports regression only")
        if training_args is None:
            training_args = {}
        if meta_args is None:
            meta_args = {}
        meta_args.setdefault('save_path', f'results/{self.base_name}')
        check_dir(meta_args['save_path'])
        self.meta_config = meta_args
        self.training_config = training_args
        training_args.setdefault('optimizer', 'adamw')
        training_args.setdefault('lr', 1e-4)
        training_args.setdefault('weight_decay', 0.0)
        training_args.setdefault('batch_size', 64)
        training_args.setdefault('max_epochs', 1000)
        training_args.setdefault('patience', patience)
        training_args.setdefault('main_loss_type', 'huber')
        training_args.setdefault('huber_delta', 1.0)
        training_args.setdefault('lgbm_early_stopping_rounds', 100)
        training_args.setdefault('log_every_n_epochs', 50)

        X_num_np = self._to_numpy(X_num)
        X_cat_np = self._to_numpy(X_cat)
        y_tensor = torch.as_tensor(ys, dtype=torch.float32)
        y_np = self._to_numpy(y_tensor)
        X_train_np = _merge_features(X_num_np, X_cat_np)

        X_val_np = y_val_np = None
        if eval_set is not None:
            val_set = eval_set[0]
            X_val_np = _merge_features(self._to_numpy(val_set[0]), self._to_numpy(val_set[1]))
            y_val_np = self._to_numpy(val_set[2])

        self._fit_or_load_lgbm(X_train_np, y_np, X_val_np, y_val_np, meta_args['save_path'], training_args)

        y_tree_train = torch.tensor(self._predict_tree(X_train_np), device=self.device)
        refine_x_train = self._prepare_refine_tensor(torch.as_tensor(X_num) if X_num is not None else None, torch.as_tensor(X_cat) if X_cat is not None else None).to(self.device)
        y_train = y_tensor.to(self.device).float().view(-1, 1)

        train_dataset = TensorDataset(refine_x_train, y_tree_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=training_args['batch_size'], shuffle=True)

        optimizer, scheduler = TabModel.make_optimizer(self.model, training_args)

        best_metric = None
        best_state = None
        best_epoch = 0
        no_improvement = 0
        total_time = 0.0

        for epoch in range(1, training_args['max_epochs'] + 1):
            self.model.train()
            epoch_loss = 0.0
            gamma_vals = []
            safe_vals = []
            spec_vals = []
            for refine_x_batch, y_tree_batch, y_batch in train_loader:
                optimizer.zero_grad()
                start = time.time()
                preds, extras = self.model(refine_x_batch, y_tree_batch)
                forward_time = time.time() - start
                if training_args['main_loss_type'] == 'mse':
                    loss = F.mse_loss(preds, y_batch)
                elif training_args['main_loss_type'] == 'huber':
                    loss = F.huber_loss(preds, y_batch, delta=training_args['huber_delta'])
                else:
                    raise ValueError(f"Unsupported main_loss_type: {training_args['main_loss_type']}")
                start = time.time()
                loss.backward()
                backward_time = time.time() - start
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_time += forward_time + backward_time
                epoch_loss += float(loss.detach().cpu().item())
                gamma_vals.append(float(extras['gamma'].detach().mean().cpu().item()))
                safe_vals.append(float(extras['delta_safe'].detach().abs().mean().cpu().item()))
                spec_vals.append(float(extras['delta_spec'].detach().abs().mean().cpu().item()))

            train_preds, train_results = self.predict(
                X_num=X_num,
                X_cat=X_cat,
                ys=ys,
                y_std=y_std,
                task=task,
                return_metric=True,
                return_loss=True,
            )
            val_results = None
            current_metric = train_results['metric'][0]
            if eval_set is not None:
                _, val_results = self.predict(
                    X_num=eval_set[0][0],
                    X_cat=eval_set[0][1],
                    ys=eval_set[0][2],
                    y_std=y_std,
                    task=task,
                    return_metric=True,
                    return_loss=True,
                )
                current_metric = val_results['metric'][0]

            self.history['train']['loss'].append(epoch_loss / max(len(train_loader), 1))
            self.history['train']['tot_time'] = total_time
            self.history['val']['metric_name'] = 'rmse'
            if val_results is not None:
                self.history['val']['metric'].append(current_metric)

            self.append_log(
                meta_args['save_path'],
                (
                    f"[epoch] {epoch} | train_rmse={train_results['metrics']['rmse']:.6g}"
                    f" | train_mae={train_results['metrics']['mae']:.6g}"
                    f" | train_r2={train_results['metrics']['r2']:.6g}"
                    + (
                        f" | val_rmse={val_results['metrics']['rmse']:.6g}"
                        f" | val_mae={val_results['metrics']['mae']:.6g}"
                        f" | val_r2={val_results['metrics']['r2']:.6g}"
                        if val_results is not None else ""
                    )
                    + (
                        f" | gamma_mean={np.mean(gamma_vals):.6g}"
                        f" | abs_delta_safe={np.mean(safe_vals):.6g}"
                        f" | abs_delta_spec={np.mean(spec_vals):.6g}"
                    )
                ),
            )

            if best_metric is None or current_metric < best_metric:
                best_metric = current_metric
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch
                no_improvement = 0
            else:
                no_improvement += 1

            if training_args['patience'] > 0 and no_improvement >= training_args['patience']:
                self.append_log(meta_args['save_path'], f"[early_stop] best_epoch={best_epoch} | stop_epoch={epoch} | best_val_rmse={best_metric:.6g}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.history['val']['best_metric'] = best_metric
        self.history['val']['best_epoch'] = best_epoch
        self.save(meta_args['save_path'])

    def predict(
        self,
        dev_loader=None,
        X_num=None,
        X_cat=None,
        ys=None,
        ids=None,
        y_std=None,
        task: str = None,
        return_probs: bool = True,
        return_metric: bool = False,
        return_loss: bool = False,
        meta_args: ty.Optional[dict] = None,
    ):
        if task != 'regression':
            raise NotImplementedError("lgbm_cgr_hybrid currently supports regression only")
        X_num_np = self._to_numpy(X_num)
        X_cat_np = self._to_numpy(X_cat)
        X_np = _merge_features(X_num_np, X_cat_np)
        y_tree_np = self._predict_tree(X_np)
        y_tree = torch.tensor(y_tree_np, device=self.device)
        refine_x = self._prepare_refine_tensor(torch.as_tensor(X_num) if X_num is not None else None, torch.as_tensor(X_cat) if X_cat is not None else None).to(self.device)
        y_true = None if ys is None else torch.as_tensor(ys).detach().cpu().numpy()

        self.model.eval()
        start = time.time()
        with torch.no_grad():
            predictions, _ = self.model(refine_x, y_tree)
        used_time = time.time() - start
        self.model.train()
        predictions = predictions.squeeze(-1).detach().cpu().numpy()

        loss = None
        metric = None
        metrics = None
        if return_loss and ys is not None:
            if task == 'regression':
                loss = float(np.mean((predictions - y_true) ** 2))
        if return_metric and ys is not None:
            metrics = TabModel.calculate_metric_details(y_true, predictions, task, None, y_std)
            metric = TabModel.calculate_metric(y_true, predictions, task, None, y_std)
        results = {
            'loss': loss,
            'metric': metric,
            'metrics': metrics,
            'time': used_time,
            'log_loss': None,
        }
        if meta_args is not None:
            self.save_prediction(meta_args['save_path'], results)
        return predictions, results

    def save(self, output_dir):
        check_dir(output_dir)
        torch.save(self.model.state_dict(), Path(output_dir) / 'final.bin')
        if self.lgbm_model is not None:
            with open(Path(output_dir) / 'lgbm_model.pkl', 'wb') as f:
                pickle.dump(self.lgbm_model, f)
        self.save_history(output_dir)
        self.save_config(output_dir)

    def load_best_dnn(self, save_path, file='best'):
        state_file = Path(save_path) / 'final.bin'
        if state_file.exists():
            self.model.load_state_dict(torch.load(state_file, map_location=self.device))
        lgbm_file = Path(save_path) / 'lgbm_model.pkl'
        if lgbm_file.exists():
            with open(lgbm_file, 'rb') as f:
                self.lgbm_model = pickle.load(f)
