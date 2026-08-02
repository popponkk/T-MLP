import inspect
import time
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .abstract import TabModel, check_dir

try:
    from tabm import TabM as _OfficialTabM
    _TABM_IMPORT_ERROR = None
except Exception as err:  # pragma: no cover - import depends on external package
    _OfficialTabM = None
    _TABM_IMPORT_ERROR = err


def tabm_regression_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    if y_true.ndim == 2 and y_true.shape[-1] == 1:
        y_true = y_true.squeeze(-1)
    if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
        y_pred = y_pred.squeeze(-1)
    if y_pred.ndim == 2 and y_true.ndim == 1:
        y_true = y_true.unsqueeze(1).expand(-1, y_pred.shape[1])
    return F.mse_loss(y_pred, y_true)


class _TabMWrapper(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: ty.Optional[int] = None,
        d_in: ty.Optional[int] = None,
        input_dim: ty.Optional[int] = None,
        d_out: int = 1,
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 512,
        dropout: float = 0.1,
        cat_cardinalities: ty.Optional[ty.Sequence[int]] = None,
        **kwargs: ty.Any,
    ) -> None:
        super().__init__()
        if _OfficialTabM is None:
            raise ImportError(
                "The official `tabm` package is required. Install it with `pip install tabm`."
            ) from _TABM_IMPORT_ERROR

        if n_num_features is None:
            n_num_features = d_in if d_in is not None else input_dim
        if n_num_features is None:
            raise ValueError(
                "TabMModel requires `n_num_features`, `d_in`, or `input_dim`."
            )

        self.n_num_features = int(n_num_features)
        self.d_out = int(d_out)
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.k = int(k)

        make_kwargs = {
            "n_num_features": self.n_num_features,
            "cat_cardinalities": self.cat_cardinalities,
            "d_out": self.d_out,
            "k": self.k,
            "n_blocks": int(n_blocks),
            "d_block": int(d_block),
            "dropout": float(dropout),
            **kwargs,
        }
        signature = inspect.signature(_OfficialTabM.make)
        supported = {
            key: value
            for key, value in make_kwargs.items()
            if key in signature.parameters
        }
        supported["d_out"] = int(d_out)
        supported["n_num_features"] = self.n_num_features
        supported["cat_cardinalities"] = self.cat_cardinalities
        supported["k"] = self.k
        supported["n_blocks"] = int(n_blocks)
        supported["d_block"] = int(d_block)
        supported["dropout"] = float(dropout)
        self.model = _OfficialTabM.make(**supported)
        # Trust the official model if it rewrites k internally.
        self.k = int(getattr(self.model, "k", self.k))

    def _unpack_inputs(
        self,
        x_num: ty.Union[torch.Tensor, dict, None],
        x_cat: ty.Optional[torch.Tensor] = None,
    ) -> tuple[ty.Optional[torch.Tensor], ty.Optional[torch.Tensor]]:
        if isinstance(x_num, dict):
            batch = x_num
            x_cat = batch.get("x_cat", batch.get("cat", x_cat))
            x_num = batch.get("x_num", batch.get("x"))
        return x_num, x_cat

    def forward(
        self,
        x_num: ty.Union[torch.Tensor, dict, None],
        x_cat: ty.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_num, x_cat = self._unpack_inputs(x_num, x_cat)
        if x_num is None:
            raise ValueError("TabMModel requires numerical features as `x_num`.")
        x_num = x_num.float()
        if self.cat_cardinalities:
            if x_cat is None:
                raise ValueError("Categorical features were configured but `x_cat` is missing.")
            y_pred = self.model(x_num, x_cat.long())
        else:
            y_pred = self.model(x_num)

        if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)
        if not self.training and y_pred.ndim >= 2:
            y_pred = y_pred.mean(dim=1)
            if y_pred.ndim == 2 and y_pred.shape[-1] == 1:
                y_pred = y_pred.squeeze(-1)
        return y_pred


class TabMModel(TabModel):
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
            raise NotImplementedError("tabm does not support sparse gating options")
        super().__init__()
        model_config = self.preproc_config(model_config)
        model_config["d_out"] = n_labels
        cat_cardinalities = [] if categories is None else list(categories)
        self.model = _TabMWrapper(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            **model_config,
        ).to(device)
        self.base_name = "tabm"
        self.device = torch.device(device)

    def preproc_config(self, model_config: dict):
        self.saved_model_config = model_config.copy()
        model_config.pop("model_name", None)
        model_config.pop("base_model", None)
        model_config.setdefault("k", 32)
        model_config.setdefault("n_blocks", 3)
        model_config.setdefault("d_block", 512)
        model_config.setdefault("dropout", 0.1)
        model_config.setdefault("d_out", 1)
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
            raise NotImplementedError("tabm currently supports regression only")

        if meta_args is None:
            meta_args = {}
        meta_args.setdefault("save_path", f"results/{self.base_name}")
        meta_args.setdefault("log_every_n_epochs", 50)
        check_dir(meta_args["save_path"])
        self.meta_config = meta_args

        training_args = {} if training_args is None else training_args
        training_args.setdefault("optimizer", "adamw")
        training_args.setdefault("weight_decay", 0.0)
        training_args.setdefault("batch_size", 64)
        training_args.setdefault("max_epochs", 10000)
        training_args.setdefault("save_frequency", "epoch")
        training_args.setdefault("patience", patience)
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
            dev_eval_set = eval_set[0]
            dev_loader = TabModel.prepare_tensor_loader(
                X_num=dev_eval_set[0],
                X_cat=dev_eval_set[1],
                ys=dev_eval_set[2],
                ids=dev_eval_set[3],
                batch_size=training_args["batch_size"],
            )
            test_loader = None
            if len(eval_set) == 2:
                test_eval_set = eval_set[1]
                test_loader = TabModel.prepare_tensor_loader(
                    X_num=test_eval_set[0],
                    X_cat=test_eval_set[1],
                    ys=test_eval_set[2],
                    ids=test_eval_set[3],
                    batch_size=training_args["batch_size"],
                )
        else:
            dev_loader, test_loader = None, None

        steps_per_epoch = len(train_loader)
        tot_step, tot_time = 0, 0.0
        for _ in range(training_args["max_epochs"]):
            self.model.train()
            tot_loss = 0.0
            for _, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x_num, x_cat, y = TabModel.parse_batch(batch, placeholders, self.device)
                start_time = time.time()
                logits = self.model(x_num, x_cat)
                forward_time = time.time() - start_time
                loss = tabm_regression_loss(logits, y)

                start_time = time.time()
                loss.backward()
                backward_time = time.time() - start_time
                self.gradient_policy()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                tot_step += 1
                tot_loss += loss.detach().cpu().item()
                tot_time += forward_time + backward_time

            if training_args["save_frequency"] == "epoch":
                is_early_stop = self.save_evaluate_dnn(
                    tot_step,
                    steps_per_epoch,
                    tot_loss,
                    tot_time,
                    task,
                    training_args["patience"],
                    meta_args["save_path"],
                    dev_loader,
                    y_std,
                    test_loader=test_loader,
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
        if task != "regression":
            raise NotImplementedError("tabm currently supports regression only")

        def inference_step(model, x_num, x_cat):
            start_time = time.time()
            logits = model(x_num, x_cat)
            used_time = time.time() - start_time
            return logits, used_time

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
        self.save_config(output_dir)
        self.save_history(output_dir)


Model = TabMModel
