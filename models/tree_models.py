import pickle
import time
import typing as ty
import inspect
from pathlib import Path

import numpy as np

from models.abstract import TabModel, check_dir


def _merge_features(x_num, x_cat):
    if x_num is None and x_cat is None:
        raise ValueError("At least one of X_num or X_cat must be provided.")
    if x_num is None:
        return x_cat
    if x_cat is None:
        return x_num
    return np.concatenate([x_num, x_cat], axis=1)


def _prepare_targets(task: str, ys: np.ndarray) -> np.ndarray:
    if task == 'multiclass' and ys.ndim == 2:
        return np.argmax(ys, axis=1)
    if task == 'binclass' and ys.max() != 1.0:
        return (ys == ys.max()).astype(np.float32)
    return ys


def _filter_supported_kwargs(func, kwargs):
    signature = inspect.signature(func)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


class _TreeModel(TabModel):
    estimator_cls = None
    model_name = None

    def __init__(
        self,
        model_config: dict,
        n_num_features: int,
        categories: ty.Optional[ty.List[int]],
        n_labels: int,
        device='cpu',
    ):
        super().__init__()
        self.saved_model_config = model_config.copy()
        self.base_name = self.model_name
        self.device = None
        self.model = self._build_estimator(model_config, n_labels)

    def _build_estimator(self, model_config: dict, n_labels: int):
        raise NotImplementedError

    def _fit_estimator(self, X_train, y_train, X_val, y_val, training_args, task):
        raise NotImplementedError

    def _predict_values(self, X, task):
        raise NotImplementedError

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
        if training_args is None:
            training_args = {}
        if meta_args is None:
            meta_args = {}
        meta_args.setdefault('save_path', f'results/{self.base_name}')
        check_dir(meta_args['save_path'])
        self.meta_config = meta_args
        self.training_config = training_args

        X_train = _merge_features(X_num, X_cat)
        y_train = _prepare_targets(task, ys)

        X_val = y_val = None
        if eval_set is not None:
            val_set = eval_set[0]
            X_val = _merge_features(val_set[0], val_set[1])
            y_val = _prepare_targets(task, val_set[2])

        start_time = time.time()
        self._fit_estimator(X_train, y_train, X_val, y_val, training_args, task)
        self.history['train']['tot_time'] = time.time() - start_time
        self.save(meta_args['save_path'])

        if X_val is not None:
            _, results = self.predict(
                X_num=val_set[0],
                X_cat=val_set[1],
                ys=val_set[2],
                y_std=y_std,
                task=task,
                return_metric=True,
                return_loss=True,
            )
            self.history['val']['metric_name'] = results['metric'][1]
            self.history['val']['metric'].append(results['metric'][0])
            self.history['val']['best_metric'] = results['metric'][0]
            self.history['val']['best_metrics'].append(results['metric'][0])

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
        X = _merge_features(X_num, X_cat)
        y_true = None if ys is None else _prepare_targets(task, ys)
        start_time = time.time()
        predictions = self._predict_values(X, task)
        used_time = time.time() - start_time

        if return_loss and y_true is not None:
            if task == 'regression':
                loss = float(np.mean((predictions - y_true) ** 2))
            else:
                loss = None
        else:
            loss = None

        if task == 'regression':
            prediction_type = None
        elif return_probs:
            prediction_type = 'probs'
        else:
            prediction_type = None

        if return_metric and y_true is not None:
            metrics = TabModel.calculate_metric_details(
                y_true, predictions, task, prediction_type, y_std
            )
            metric = TabModel.calculate_metric(
                y_true, predictions, task, prediction_type, y_std
            )
        else:
            metrics = None
            metric = None

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
        with open(Path(output_dir) / 'final.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        self.save_history(output_dir)
        self.save_config(output_dir)

    def load_best_dnn(self, save_path, file='best'):
        model_file = Path(save_path) / 'final.pkl'
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)


class XGBoostModel(_TreeModel):
    model_name = 'xgboost'

    def _build_estimator(self, model_config: dict, n_labels: int):
        from xgboost import XGBClassifier, XGBRegressor

        self._regressor_cls = XGBRegressor
        self._classifier_cls = XGBClassifier
        self.model_config = model_config.copy()
        return None

    def _ensure_model(self, task):
        if self.model is not None:
            return
        if task == 'regression':
            self.model = self._regressor_cls(**self.model_config)
        else:
            extra = {'disable_default_eval_metric': True}
            if task == 'multiclass':
                extra['objective'] = 'multi:softprob'
            self.model = self._classifier_cls(**self.model_config, **extra)

    def _fit_estimator(self, X_train, y_train, X_val, y_val, training_args, task):
        self._ensure_model(task)
        fit_kwargs = training_args.copy()
        if X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
        fit_kwargs = _filter_supported_kwargs(self.model.fit, fit_kwargs)
        self.model.fit(X_train, y_train, **fit_kwargs)

    def _predict_values(self, X, task):
        if task == 'regression':
            return self.model.predict(X)
        if task == 'binclass':
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)


class CatBoostModel(_TreeModel):
    model_name = 'catboost'

    def _build_estimator(self, model_config: dict, n_labels: int):
        from catboost import CatBoostClassifier, CatBoostRegressor

        self._regressor_cls = CatBoostRegressor
        self._classifier_cls = CatBoostClassifier
        self.model_config = model_config.copy()
        return None

    def _ensure_model(self, task):
        if self.model is not None:
            return
        if task == 'regression':
            self.model = self._regressor_cls(**self.model_config)
        else:
            self.model = self._classifier_cls(**self.model_config)

    def _fit_estimator(self, X_train, y_train, X_val, y_val, training_args, task):
        self._ensure_model(task)
        fit_kwargs = training_args.copy()
        if X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = (X_val, y_val)
        fit_kwargs = _filter_supported_kwargs(self.model.fit, fit_kwargs)
        self.model.fit(X_train, y_train, **fit_kwargs)

    def _predict_values(self, X, task):
        if task == 'regression':
            return self.model.predict(X)
        if task == 'binclass':
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)


class LightGBMModel(_TreeModel):
    model_name = 'lightgbm'

    def _build_estimator(self, model_config: dict, n_labels: int):
        import lightgbm as lgb

        self._lgb = lgb
        self.model_config = model_config.copy()
        return None

    def _ensure_model(self, task):
        if self.model is not None:
            return
        if task == 'regression':
            self.model = self._lgb.LGBMRegressor(**self.model_config)
        else:
            self.model = self._lgb.LGBMClassifier(**self.model_config)

    def _fit_estimator(self, X_train, y_train, X_val, y_val, training_args, task):
        self._ensure_model(task)
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            if 'early_stopping_rounds' in training_args:
                fit_kwargs['callbacks'] = [
                    self._lgb.early_stopping(training_args['early_stopping_rounds'], verbose=False),
                    self._lgb.log_evaluation(0),
                ]
        fit_kwargs = _filter_supported_kwargs(self.model.fit, fit_kwargs)
        self.model.fit(X_train, y_train, **fit_kwargs)

    def _predict_values(self, X, task):
        if task == 'regression':
            return self.model.predict(X)
        if task == 'binclass':
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)
