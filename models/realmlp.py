import inspect
import typing as ty

import numpy as np

from .tree_models import _TreeModel, _filter_supported_kwargs

try:
    from pytabkit import RealMLP_TD_Regressor as _RealMLPRegressor
    _REALMLP_IMPORT_ERROR = None
except Exception:
    try:
        from pytabkit.models.sklearn.sklearn_interfaces import (
            RealMLP_TD_Regressor as _RealMLPRegressor,
        )
        _REALMLP_IMPORT_ERROR = None
    except Exception as err:  # pragma: no cover - depends on external package
        _RealMLPRegressor = None
        _REALMLP_IMPORT_ERROR = err


def _to_numpy_float32(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _to_numpy_target(y):
    if y is None:
        return None
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    return np.asarray(y).reshape(-1)


class RealMLPBaseline:
    def __init__(
        self,
        *,
        device: str = "cuda",
        random_state: int = 42,
        n_epochs: int = 256,
        batch_size: int = 32,
        lr: float = 1e-5,
        **kwargs,
    ) -> None:
        if _RealMLPRegressor is None:
            raise ImportError(
                "RealMLP requires `pytabkit[models]`. Install it with "
                '`pip install "pytabkit[models]"`.'
            ) from _REALMLP_IMPORT_ERROR

        self.model_config = {
            "device": device,
            "random_state": random_state,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            **kwargs,
        }
        signature = inspect.signature(_RealMLPRegressor)
        supported = _filter_supported_kwargs(_RealMLPRegressor, self.model_config)
        self.model = _RealMLPRegressor(**supported)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = _to_numpy_float32(X_train)
        y_train = _to_numpy_target(y_train)
        X_val = _to_numpy_float32(X_val)
        y_val = _to_numpy_target(y_val)

        fit_kwargs = {}
        fit_signature = inspect.signature(self.model.fit)
        common_val_kwargs = {
            "X_val": X_val,
            "y_val": y_val,
            "eval_set": None if X_val is None or y_val is None else [(X_val, y_val)],
            "validation_data": None if X_val is None or y_val is None else (X_val, y_val),
        }
        for key, value in common_val_kwargs.items():
            if value is not None and key in fit_signature.parameters:
                fit_kwargs[key] = value
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X):
        X = _to_numpy_float32(X)
        y_pred = self.model.predict(X)
        if hasattr(y_pred, "detach"):
            y_pred = y_pred.detach().cpu().numpy()
        return np.asarray(y_pred).reshape(-1)


class RealMLPModel(_TreeModel):
    model_name = "realmlp"
    estimator_cls = RealMLPBaseline

    def _build_estimator(self, model_config: dict, n_labels: int):
        if n_labels != 1:
            raise NotImplementedError("realmlp currently supports scalar regression only")
        self.model_config = model_config.copy()
        return self.estimator_cls(**self.model_config)

    def _fit_estimator(self, X_train, y_train, X_val, y_val, training_args, task):
        if task != "regression":
            raise NotImplementedError("realmlp currently supports regression only")
        self.model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    def _predict_values(self, X, task):
        if task != "regression":
            raise NotImplementedError("realmlp currently supports regression only")
        return self.model.predict(X)


Model = RealMLPBaseline
