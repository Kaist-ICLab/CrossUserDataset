"""pytorch-widedeep wrapper covering SAINT, TabTransformer, FTTransformer.

This file owns the linear-attention monkey patches the wrapper installs
on the upstream SAINT / FT encoder blocks; the patches are idempotent
(guarded by a `_ubicomp_linear_*_patch` attribute on the encoder
module) and therefore safe to import multiple times.
"""

import copy
import inspect
import logging

import einops
import numpy as np
import pandas as pd
import pytorch_widedeep.models.tabular.transformers.ft_transformer as _ft_mod
import pytorch_widedeep.models.tabular.transformers.saint as _saint_mod
import torch
import torch.nn as nn
from pytorch_widedeep.callbacks import EarlyStopping as WideEarlyStopping
from pytorch_widedeep.metrics import Accuracy as WideAccuracy
from pytorch_widedeep.models import SAINT
from pytorch_widedeep.models import FTTransformer as WideFTTransformer
from pytorch_widedeep.models import TabTransformer, WideDeep
from pytorch_widedeep.models.tabular.transformers._attention_layers import (
    AddNorm,
    FeedForward,
    MultiHeadedAttention,
    NormAdd,
)
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.training import Trainer as WideTrainer
from pytorch_widedeep.training._wd_dataset import WideDeepDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from torchmetrics.classification import BinaryAUROC

logger = logging.getLogger(__name__)

from ._helpers import (
    FIXED_BATCH_SIZE,
    _drop_optuna_helper_params,
    _filter_supported_kwargs,
    attach_training_metadata,
)


class WidedeepWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_type="SAINT",
        batch_size=FIXED_BATCH_SIZE,
        epochs=50,
        patience=20,
        efficient_attention=False,
        **kwargs,
    ):
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.efficient_attention = efficient_attention
        self.kwargs = kwargs
        self.trainer = None
        self.preprocessor = None
        self.col_names = None

    def fit(self, X, y, X_val=None, y_val=None):
        training_params = _drop_optuna_helper_params(self.kwargs)
        train_lr = float(training_params.get("lr", 1e-3))
        train_weight_decay = float(training_params.get("weight_decay", 0.0) or 0.0)
        effective_model_params = {}
        self.batch_size = int(self.batch_size or FIXED_BATCH_SIZE)

        # Convert to DataFrame
        self.col_names = [f"col_{i}" for i in range(X.shape[1])]
        df_train = pd.DataFrame(X, columns=self.col_names)
        df_train["target"] = y

        # Preprocessing
        self.preprocessor = TabPreprocessor(
            continuous_cols=self.col_names, scale=False
        )  # Already scaled
        X_tab = self.preprocessor.fit_transform(df_train)

        # Validation
        if X_val is not None:
            df_val = pd.DataFrame(X_val, columns=self.col_names)
            df_val["target"] = y_val
            X_val_tab = self.preprocessor.transform(df_val)
        else:
            X_val_tab = None
            df_val = None

        def _cap_transformer_params(
            params,
            model_name,
            max_input_dim=32,
            max_heads=4,
            max_blocks=3,
            token_element_budget=24_000_000,
        ):
            params.setdefault("input_dim", max_input_dim)
            params.setdefault("n_heads", min(4, max_heads))
            params.setdefault("n_blocks", 2)

            input_dim = min(int(params.get("input_dim", max_input_dim)), max_input_dim)
            n_heads = min(max(1, int(params.get("n_heads", 4))), max_heads, input_dim)
            n_blocks = min(max(1, int(params.get("n_blocks", 2))), max_blocks)

            if input_dim % n_heads != 0:
                input_dim = max(n_heads, (input_dim // n_heads) * n_heads)

            n_features = max(1, len(self.col_names))
            max_dim = max(
                n_heads, token_element_budget // max(1, n_features * max(1, self.batch_size))
            )
            max_dim = max(n_heads, (max_dim // n_heads) * n_heads)
            if input_dim > max_dim:
                logger.info(
                    f"[INFO] {model_name}: capped input_dim from {params.get('input_dim')} to {max_dim} "
                    f"for {n_features} features and batch_size={self.batch_size}."
                )
                input_dim = max_dim

            max_batch = max(1, token_element_budget // max(1, n_features * max(1, input_dim)))
            if self.batch_size > max_batch:
                adjusted_batch = next(
                    (candidate for candidate in (64, 32, 16, 8) if candidate <= max_batch), 8
                )
                logger.info(
                    f"[INFO] {model_name}: capped batch_size from {self.batch_size} to {adjusted_batch} "
                    f"for {n_features} features and input_dim={input_dim}."
                )
                self.batch_size = adjusted_batch

            params["input_dim"] = input_dim
            params["n_heads"] = n_heads
            params["n_blocks"] = n_blocks

        # Define Model
        if self.model_type == "SAINT":
            saint_params = training_params.copy()
            _TRAIN_KEYS = {
                "lr",
                "weight_decay",
                "batch_size",
                "mlp_dropout",
                "num_layers",
                "hidden_dim",
                "transformer_dropout",
            }
            for k in _TRAIN_KEYS:
                saint_params.pop(k, None)
            if "dropout" in saint_params:
                dropout_val = saint_params.pop("dropout")
                saint_params.setdefault("attn_dropout", dropout_val)
                saint_params.setdefault("ff_dropout", dropout_val)
            _cap_transformer_params(saint_params, "SAINT")
            if self.efficient_attention:
                logger.info(
                    "[INFO] SAINT: enabling linear-attention path for SAINT encoder blocks."
                )
                if not getattr(_saint_mod, "_ubicomp_linear_saint_patch", False):

                    class EfficientSaintEncoder(nn.Module):
                        def __init__(
                            self,
                            input_dim: int,
                            n_heads: int,
                            use_bias: bool,
                            attn_dropout: float,
                            ff_dropout: float,
                            ff_factor: int,
                            activation: str,
                            n_feat: int,
                        ):
                            super().__init__()
                            self.n_feat = n_feat

                            self.col_attn = MultiHeadedAttention(
                                input_dim,
                                n_heads,
                                use_bias,
                                attn_dropout,
                                None,
                                True,  # use_linear_attention
                                False,  # use_flash_attention
                            )
                            self.col_attn_ff = FeedForward(
                                input_dim, ff_dropout, ff_factor, activation
                            )
                            self.col_attn_addnorm = AddNorm(input_dim, attn_dropout)
                            self.col_attn_ff_addnorm = AddNorm(input_dim, ff_dropout)

                            # Row attention over samples per feature token (keeps SAINT row-attention intent,
                            # avoids flattening n_feat * input_dim into huge projection matrices).
                            self.row_attn = MultiHeadedAttention(
                                input_dim,
                                n_heads,
                                use_bias,
                                attn_dropout,
                                None,
                                True,  # use_linear_attention
                                False,  # use_flash_attention
                            )
                            self.row_attn_ff = FeedForward(
                                input_dim, ff_dropout, ff_factor, activation
                            )
                            self.row_attn_addnorm = AddNorm(input_dim, attn_dropout)
                            self.row_attn_ff_addnorm = AddNorm(input_dim, ff_dropout)

                        def forward(self, X):
                            x = self.col_attn_addnorm(X, self.col_attn)
                            x = self.col_attn_ff_addnorm(x, self.col_attn_ff)
                            x = einops.rearrange(x, "b n d -> n b d")
                            x = self.row_attn_addnorm(x, self.row_attn)
                            x = self.row_attn_ff_addnorm(x, self.row_attn_ff)
                            x = einops.rearrange(x, "n b d -> b n d")
                            return x

                    _saint_mod.SaintEncoder = EfficientSaintEncoder
                    _saint_mod._ubicomp_linear_saint_patch = True

            saint_params = _filter_supported_kwargs(SAINT.__init__, saint_params)
            effective_model_params = dict(saint_params)
            deeptabular = SAINT(
                column_idx=self.preprocessor.column_idx,
                continuous_cols=self.col_names,
                **saint_params,
            )

        elif self.model_type == "TabTransformer":
            tt_params = training_params.copy()
            _TRAIN_KEYS = {
                "lr",
                "weight_decay",
                "batch_size",
                "mlp_dropout",
                "num_layers",
                "hidden_dim",
                "transformer_dropout",
                "miscellaneous_dropout",
            }
            for k in _TRAIN_KEYS:
                tt_params.pop(k, None)
            if "dropout" in tt_params:
                dropout_val = tt_params.pop("dropout")
                tt_params.setdefault("attn_dropout", dropout_val)
                tt_params.setdefault("ff_dropout", dropout_val)
            tt_params.setdefault("input_dim", 32)
            tt_params.setdefault("n_heads", 4)
            tt_params.setdefault("n_blocks", 2)
            _cap_transformer_params(tt_params, "TabTransformer")
            tt_params["use_linear_attention"] = bool(self.efficient_attention)
            if self.efficient_attention:
                logger.info(
                    "[INFO] TabTransformer: use_linear_attention=True (architecture preserved)."
                )
            tt_params = _filter_supported_kwargs(TabTransformer.__init__, tt_params)
            effective_model_params = dict(tt_params)
            deeptabular = TabTransformer(
                column_idx=self.preprocessor.column_idx,
                continuous_cols=self.col_names,
                embed_continuous_method="standard",
                **tt_params,
            )

        elif self.model_type == "FTTransformer":
            ft_params = training_params.copy()
            _TRAIN_KEYS = {
                "lr",
                "weight_decay",
                "batch_size",
                "mlp_dropout",
                "num_layers",
                "hidden_dim",
            }
            for k in _TRAIN_KEYS:
                ft_params.pop(k, None)
            if "dropout" in ft_params:
                dropout_val = ft_params.pop("dropout")
                ft_params.setdefault("attn_dropout", dropout_val)
                ft_params.setdefault("ff_dropout", dropout_val)
            ft_params.setdefault("input_dim", 32)
            ft_params.setdefault("n_heads", 4)
            if int(ft_params.get("n_blocks", 2)) > 3:
                logger.info(
                    f"[INFO] FTTransformer: capped n_blocks from {ft_params.get('n_blocks')} to 3."
                )
                ft_params["n_blocks"] = 3
            # Multi-head attention requires the embedding width to be divisible by the number of heads.
            # Coerce misconfigured HPO suggestions to the nearest valid multiple instead of failing late.
            input_dim = int(ft_params.get("input_dim", 32))
            n_heads = int(ft_params.get("n_heads", 4))
            if n_heads <= 0:
                n_heads = 4
            if input_dim % n_heads != 0:
                adjusted = max(n_heads, int(round(input_dim / n_heads)) * n_heads)
                if adjusted != input_dim:
                    logger.info(
                        f"[INFO] FTTransformer: adjusted input_dim from {input_dim} to {adjusted} for n_heads={n_heads}."
                    )
                ft_params["input_dim"] = adjusted
                ft_params["n_heads"] = n_heads
                input_dim = adjusted

            max_ft_input_dim = 64
            if input_dim > max_ft_input_dim:
                adjusted = max(n_heads, (max_ft_input_dim // n_heads) * n_heads)
                logger.info(
                    f"[INFO] FTTransformer: capped input_dim from {input_dim} to {adjusted}."
                )
                ft_params["input_dim"] = adjusted
                input_dim = adjusted

            n_features = max(1, len(self.col_names))
            token_element_budget = 36_000_000
            max_dim = max(
                n_heads, token_element_budget // max(1, n_features * max(1, self.batch_size))
            )
            max_dim = max(n_heads, (max_dim // n_heads) * n_heads)
            if input_dim > max_dim:
                logger.info(
                    f"[INFO] FTTransformer: capped input_dim from {input_dim} to {max_dim} "
                    f"for {n_features} features and batch_size={self.batch_size}."
                )
                ft_params["input_dim"] = max_dim
                input_dim = max_dim

            max_batch = max(1, token_element_budget // max(1, n_features * max(1, input_dim)))
            if self.batch_size > max_batch:
                adjusted_batch = next(
                    (candidate for candidate in (64, 32, 16, 8) if candidate <= max_batch), 8
                )
                logger.info(
                    f"[INFO] FTTransformer: capped batch_size from {self.batch_size} to {adjusted_batch} "
                    f"for {n_features} features and input_dim={input_dim}."
                )
                self.batch_size = adjusted_batch
            if self.efficient_attention:
                logger.info(
                    "[INFO] FTTransformer: enabling kernel linear-attention path in FT encoder blocks."
                )
                if not getattr(_ft_mod, "_ubicomp_linear_ft_patch", False):

                    class EfficientFTTransformerEncoder(nn.Module):
                        def __init__(
                            self,
                            input_dim: int,
                            n_feats: int,
                            n_heads: int,
                            use_bias: bool,
                            attn_dropout: float,
                            ff_dropout: float,
                            ff_factor: float,
                            kv_compression_factor: float,
                            kv_sharing: bool,
                            activation: str,
                            first_block: bool,
                        ):
                            super().__init__()
                            self.first_block = first_block
                            self.attn = MultiHeadedAttention(
                                input_dim,
                                n_heads,
                                use_bias,
                                attn_dropout,
                                None,
                                True,  # use_linear_attention
                                False,  # use_flash_attention
                            )
                            self.ff = FeedForward(input_dim, ff_dropout, ff_factor, activation)
                            self.attn_normadd = NormAdd(input_dim, attn_dropout)
                            self.ff_normadd = NormAdd(input_dim, ff_dropout)

                        def forward(self, X):
                            if self.first_block:
                                x = X + self.attn(X)
                            else:
                                x = self.attn_normadd(X, self.attn)
                            return self.ff_normadd(x, self.ff)

                    _ft_mod.FTTransformerEncoder = EfficientFTTransformerEncoder
                    _ft_mod._ubicomp_linear_ft_patch = True

            ft_params = _filter_supported_kwargs(WideFTTransformer.__init__, ft_params)
            effective_model_params = dict(ft_params)
            deeptabular = WideFTTransformer(
                column_idx=self.preprocessor.column_idx,
                continuous_cols=self.col_names,
                embed_continuous_method="standard",
                **ft_params,
            )

        model = WideDeep(deeptabular=deeptabular)

        # Trainer
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Callbacks
        callbacks = []
        # Use ROCAUC for early stopping if possible, or Accuracy
        if self.patience > 0:
            # Torchmetrics names are logged as val_<ClassName>, e.g. val_BinaryAUROC.
            callbacks.append(
                WideEarlyStopping(
                    monitor="val_BinaryAUROC",
                    mode="max",
                    patience=self.patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                )
            )

        # Keep Accuracy for visibility and add BinaryAUROC so callbacks can monitor val_BinaryAUROC.
        self.trainer = WideTrainer(
            model,
            objective="binary",
            optimizers=torch.optim.Adam(
                model.parameters(), lr=train_lr, weight_decay=train_weight_decay
            ),
            metrics=[WideAccuracy, BinaryAUROC()],
            callbacks=callbacks,
            device=device,
            verbose=0,
            num_workers=0,
        )

        # Defensive Patch: WideDeepDataset sometimes receives X_tab as a dict {'X_tab': array} despite correct usage.
        # This patch unwraps it automatically.
        _original_wd_init = WideDeepDataset.__init__

        def _patched_wd_init(
            self, X_wide=None, X_tab=None, X_text=None, X_img=None, target=None, transforms=None
        ):
            if X_tab is not None and isinstance(X_tab, dict) and "X_tab" in X_tab:
                # logger.info("[DEBUG] Patch triggered: Unwrapping X_tab from dict")
                X_tab = X_tab["X_tab"]
            _original_wd_init(
                self,
                X_wide=X_wide,
                X_tab=X_tab,
                X_text=X_text,
                X_img=X_img,
                target=target,
                transforms=transforms,
            )

        WideDeepDataset.__init__ = _patched_wd_init

        X_train_dict = {"X_tab": X_tab, "target": df_train["target"].values}
        X_val_dict = (
            {"X_tab": X_val_tab, "target": df_val["target"].values}
            if X_val_tab is not None
            else None
        )

        self.trainer.fit(
            X_train=X_train_dict,
            target=None,
            X_val=X_val_dict,
            target_val=None,
            n_epochs=self.epochs,
            batch_size=self.batch_size,
        )
        history = getattr(self.trainer, "history", None)
        best_epoch = getattr(self.trainer, "best_epoch", None)
        attach_training_metadata(
            self,
            optimizer="Adam",
            best_epoch=best_epoch,
            epochs_ran=self.epochs,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            lr=train_lr,
            weight_decay=train_weight_decay,
            early_stopped=bool(
                best_epoch is not None and isinstance(best_epoch, int) and best_epoch < self.epochs
            ),
            model_selection_metric="val_BinaryAUROC",
            epoch_history=history,
            architecture_params=effective_model_params,
        )
        return self

    def predict(self, X):
        df = pd.DataFrame(X, columns=self.col_names)
        X_tab = self.preprocessor.transform(df)
        return self.trainer.predict(X_tab={"X_tab": X_tab})

    def predict_proba(self, X):
        df = pd.DataFrame(X, columns=self.col_names)
        X_tab = self.preprocessor.transform(df)
        return self.trainer.predict_proba(X_tab={"X_tab": X_tab})
