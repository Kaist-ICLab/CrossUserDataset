"""DeepCTR-Torch wrappers for DCN and AutoInt."""

import logging

import numpy as np
import torch
import torch.nn as nn
from deepctr_torch.inputs import DenseFeat, SparseFeat
from deepctr_torch.models import DCN as DeepCTRDCN
from deepctr_torch.models import AutoInt as DeepCTRAutoInt
from deepctr_torch.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)

from ._helpers import (
    FIXED_BATCH_SIZE,
    _drop_optuna_helper_params,
    _filter_supported_kwargs,
    attach_training_metadata,
)


class DeepCTRWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self, model_type="DCN", batch_size=FIXED_BATCH_SIZE, epochs=50, patience=20, **kwargs
    ):
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.kwargs = kwargs
        self.model = None
        self.feature_columns = None
        self.feature_names = None
        self._autoint_bin_edges = None

    def _fit_autoint_input(self, X, feature_names, n_bins):
        bin_edges = []
        encoded_columns = []
        feature_columns = []

        for i, name in enumerate(feature_names):
            col = X[:, i].astype(np.float32)
            qs = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.quantile(col, qs)
            edges = np.unique(edges)
            if edges.size <= 1:
                edges = np.array([edges[0], edges[0] + 1.0], dtype=np.float32)
            binned = np.digitize(col, edges[1:-1], right=False).astype(np.int64)
            vocab_size = int(max(2, edges.size))
            feature_columns.append(SparseFeat(name, vocabulary_size=vocab_size, embedding_dim=8))
            encoded_columns.append(binned)
            bin_edges.append(edges)

        self._autoint_bin_edges = bin_edges
        model_input = np.ascontiguousarray(np.stack(encoded_columns, axis=1))
        return feature_columns, [model_input]

    def _transform_autoint_input(self, X, feature_names):
        encoded_columns = []
        for i, name in enumerate(feature_names):
            col = X[:, i].astype(np.float32)
            edges = self._autoint_bin_edges[i]
            encoded_columns.append(np.digitize(col, edges[1:-1], right=False).astype(np.int64))
        return [np.ascontiguousarray(np.stack(encoded_columns, axis=1))]

    def _transform_dense_input(self, X):
        X_array = np.asarray(X, dtype=np.float32)
        if not X_array.flags.c_contiguous:
            X_array = np.ascontiguousarray(X_array)
        return [X_array]

    def fit(self, X, y, X_val=None, y_val=None):
        params = _drop_optuna_helper_params(self.kwargs)
        lr = float(params.pop("lr", 1e-3))
        weight_decay = float(params.pop("weight_decay", 0.0) or 0.0)
        effective_model_params = {}
        self.batch_size = int(self.batch_size or FIXED_BATCH_SIZE)

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        self.feature_names = feature_names
        train_model_input = self._transform_dense_input(X)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_type == "DCN":
            dcn_params = params.copy()
            if "n_cross_layers" in dcn_params and "cross_num" not in dcn_params:
                dcn_params["cross_num"] = dcn_params.pop("n_cross_layers")
            if "hidden_dropout" in dcn_params and "dnn_dropout" not in dcn_params:
                dcn_params["dnn_dropout"] = dcn_params.pop("hidden_dropout")
            if "dropout" in dcn_params and "dnn_dropout" not in dcn_params:
                dcn_params["dnn_dropout"] = dcn_params.pop("dropout")
            if "dnn_hidden_units" in dcn_params and not isinstance(
                dcn_params["dnn_hidden_units"], tuple
            ):
                dcn_params["dnn_hidden_units"] = tuple(dcn_params["dnn_hidden_units"])
            if "dnn_hidden_units" not in dcn_params and {"layer_size", "n_hidden_layers"} <= set(
                dcn_params
            ):
                dcn_params["dnn_hidden_units"] = tuple(
                    [int(dcn_params.pop("layer_size"))] * int(dcn_params.pop("n_hidden_layers"))
                )
            if "dnn_hidden_units" in dcn_params:
                original_units = tuple(int(v) for v in dcn_params["dnn_hidden_units"])
                capped_units = tuple(min(v, 256) for v in original_units[:4])
                if capped_units != original_units:
                    logger.info(
                        f"[INFO] DCN: capped dnn_hidden_units from {original_units} to {capped_units}."
                    )
                dcn_params["dnn_hidden_units"] = capped_units
            if "cross_num" in dcn_params and int(dcn_params["cross_num"]) > 4:
                logger.info(f"[INFO] DCN: capped cross_num from {dcn_params['cross_num']} to 4.")
                dcn_params["cross_num"] = 4
            dcn_params.pop("cross_dropout", None)
            dcn_params.pop("layer_size", None)
            dcn_params.pop("n_hidden_layers", None)
            dcn_params.setdefault("l2_reg_linear", weight_decay)
            dcn_params.setdefault("l2_reg_embedding", weight_decay)
            dcn_params.setdefault("l2_reg_cross", weight_decay)
            dcn_params.setdefault("l2_reg_dnn", weight_decay)
            # Keep the same raw dense input, but represent it as one wide DenseFeat to reduce
            # DeepCTR feature-index bookkeeping overhead for 9k+ column tables.
            self.feature_columns = [DenseFeat("dense_input", X.shape[1])]
            dcn_params = _filter_supported_kwargs(DeepCTRDCN.__init__, dcn_params)
            effective_model_params = dict(dcn_params)
            self.model = DeepCTRDCN(
                self.feature_columns,
                self.feature_columns,
                task="binary",
                device=device,
                **dcn_params,
            )
        elif self.model_type == "AutoInt":
            autoint_params = params.copy()
            n_bins = int(autoint_params.pop("autoint_bins", 16))
            if n_bins > 16:
                logger.info(f"[INFO] AutoInt: capped autoint_bins from {n_bins} to 16.")
                n_bins = 16
            if "dropout" in autoint_params and "dnn_dropout" not in autoint_params:
                autoint_params["dnn_dropout"] = autoint_params.pop("dropout")
            if "att_layer_num" in autoint_params and int(autoint_params["att_layer_num"]) > 3:
                logger.info(
                    f"[INFO] AutoInt: capped att_layer_num from {autoint_params['att_layer_num']} to 3."
                )
                autoint_params["att_layer_num"] = 3
            if "att_head_num" in autoint_params and int(autoint_params["att_head_num"]) > 4:
                logger.info(
                    f"[INFO] AutoInt: capped att_head_num from {autoint_params['att_head_num']} to 4."
                )
                autoint_params["att_head_num"] = 4
            autoint_params.setdefault("l2_reg_dnn", weight_decay)
            autoint_params.setdefault("l2_reg_embedding", weight_decay)
            autoint_params.pop("att_embedding_dim", None)
            self.feature_columns, train_model_input = self._fit_autoint_input(
                X, feature_names, n_bins=n_bins
            )
            autoint_params = _filter_supported_kwargs(DeepCTRAutoInt.__init__, autoint_params)
            effective_model_params = dict(autoint_params)
            self.model = DeepCTRAutoInt(
                self.feature_columns,
                self.feature_columns,
                task="binary",
                device=device,
                **autoint_params,
            )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model.compile(optimizer, "binary_crossentropy", metrics=["binary_crossentropy", "auc"])

        val_data = None
        callbacks = []
        early_stopping = None
        if X_val is not None:
            if self.model_type == "AutoInt":
                val_model_input = self._transform_autoint_input(X_val, feature_names)
            else:
                val_model_input = self._transform_dense_input(X_val)
            val_data = (val_model_input, y_val)
            if self.patience > 0:
                early_stopping = EarlyStopping(
                    monitor="val_auc",
                    min_delta=1e-4,
                    patience=self.patience,
                    verbose=1,
                    mode="max",
                    restore_best_weights=True,
                )
                callbacks.append(early_stopping)

        history = self.model.fit(
            train_model_input,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=0,
        )
        history_dict = getattr(history, "history", {}) or {}
        loss_history = history_dict.get("loss", [])
        epochs_ran = len(loss_history)
        
        # we have to determine 'best_epoch' by finding the index of the max val_auc ourselves
        best_epoch = None
        early_stopped = False
        
        if self.patience > 0 and early_stopping is not None:
            if epochs_ran < self.epochs:
                early_stopped = True
            
            val_auc = history_dict.get("val_auc", [])
            if val_auc:
                best_epoch = int(np.argmax(val_auc))

        attach_training_metadata(
            self,
            optimizer="Adam",
            best_epoch=best_epoch,
            epochs_ran=epochs_ran,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            lr=lr,
            weight_decay=weight_decay,
            early_stopped=early_stopped,
            model_selection_metric="val_auroc",
            epoch_history=history_dict,
            architecture_params=effective_model_params,
        )
        return self

    def predict(self, X):
        feature_names = self.feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        if self.model_type == "AutoInt":
            test_model_input = self._transform_autoint_input(X, feature_names)
        else:
            test_model_input = self._transform_dense_input(X)
        pred_ans = self.model.predict(test_model_input, batch_size=self.batch_size)
        return np.where(pred_ans > 0.5, 1, 0).astype(int).flatten()

    def predict_proba(self, X):
        feature_names = self.feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        if self.model_type == "AutoInt":
            test_model_input = self._transform_autoint_input(X, feature_names)
        else:
            test_model_input = self._transform_dense_input(X)
        pred_prob = self.model.predict(test_model_input, batch_size=self.batch_size)
        # Construct [p0, p1]
        return np.hstack([1 - pred_prob, pred_prob])


# --- Deep Learning Models ---
