"""LightGBM training pipeline for domain adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional

import numpy as np

try:
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

from .common import ArrayDataset, safe_accuracy, safe_auc, safe_auprc


@dataclass
class LightGBMConfig:
    pretrain_rounds: int = 512
    finetune_rounds: int = 256
    adapt_rounds: int = 128
    learning_rate: float = 0.05
    num_leaves: int = 64
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    min_child_samples: int = 20
    num_threads: int = 4
    early_stopping_rounds: int = 50


@dataclass
class LightGBMRunResult:
    train_auroc: float
    val_auroc: float
    test_auroc: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_auprc: float
    val_auprc: float
    test_auprc: float
    pretrain_val_auroc: float
    pretrain_val_accuracy: float
    pretrain_val_auprc: float
    stage_durations: Dict[str, float]
    best_iteration: Optional[int]
    booster: Optional["lgb.Booster"]


class LightGBMPipeline:
    def __init__(self, config: LightGBMConfig):
        if lgb is None:
            raise ImportError("lightgbm is required to run LightGBMPipeline")
        self.config = config

    def _stage_params(self) -> dict:
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "min_child_samples": self.config.min_child_samples,
            "verbose": -1,
            "num_threads": self.config.num_threads,
        }

    def _train_stage(
        self,
        data: ArrayDataset,
        val_data: Optional[ArrayDataset],
        num_rounds: int,
        init_model: Optional["lgb.Booster"] = None,
    ) -> Optional["lgb.Booster"]:
        if num_rounds <= 0 or data.X.size == 0:
            return init_model

        train_dataset = lgb.Dataset(data.X, label=data.y)

        callbacks = [lgb.log_evaluation(0)]
        valid_sets = []
        valid_names = []
        if (
            val_data is not None
            and val_data.X.size > 0
            and len(np.unique(val_data.y)) >= 2
        ):
            val_dataset = lgb.Dataset(val_data.X, label=val_data.y, reference=train_dataset)
            valid_sets.append(val_dataset)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds, verbose=False))

        booster = lgb.train(
            self._stage_params(),
            train_dataset,
            num_boost_round=num_rounds,
            valid_sets=valid_sets if valid_sets else None,
            valid_names=valid_names if valid_names else None,
            init_model=init_model,
            callbacks=callbacks,
        )
        return booster

    def run(
        self,
        *,
        pretrain: Optional[ArrayDataset],
        pretrain_val: Optional[ArrayDataset],
        train: ArrayDataset,
        val: ArrayDataset,
        adapt: Optional[ArrayDataset],
        evaluation: ArrayDataset,
    ) -> LightGBMRunResult:
        booster: Optional["lgb.Booster"] = None
        stage_durations: Dict[str, float] = {}
        pretrain_val_auroc = float("nan")
        pretrain_val_accuracy = float("nan")
        pretrain_val_auprc = float("nan")

        if pretrain is not None and pretrain.X.size > 0:
            start = perf_counter()
            booster = self._train_stage(pretrain, pretrain_val, self.config.pretrain_rounds, None)
            stage_durations["pretrain_seconds"] = perf_counter() - start
            if (
                pretrain_val is not None
                and pretrain_val.X.size > 0
                and booster is not None
            ):
                pretrain_val_pred = booster.predict(pretrain_val.X)
                pretrain_val_auroc = safe_auc(pretrain_val.y, pretrain_val_pred)
                pretrain_val_accuracy = safe_accuracy(pretrain_val.y, pretrain_val_pred)
                pretrain_val_auprc = safe_auprc(pretrain_val.y, pretrain_val_pred)

        start = perf_counter()
        booster = self._train_stage(train, val, self.config.finetune_rounds, booster)
        stage_durations["finetune_seconds"] = perf_counter() - start

        if adapt is not None and adapt.X.size > 0:
            start = perf_counter()
            booster = self._train_stage(adapt, val, self.config.adapt_rounds, booster)
            stage_durations["adapt_seconds"] = perf_counter() - start

        if booster is None:
            raise RuntimeError("LightGBM training did not produce a booster")

        train_pred = booster.predict(train.X)
        val_pred = booster.predict(val.X) if val.X.size > 0 else np.array([])
        test_pred = booster.predict(evaluation.X)

        train_auroc = safe_auc(train.y, train_pred)
        val_auroc = safe_auc(val.y, val_pred) if val_pred.size else float("nan")
        test_auroc = safe_auc(evaluation.y, test_pred)

        best_iteration = getattr(booster, "best_iteration", None)
        if not best_iteration:
            try:
                best_iteration = booster.current_iteration()
            except AttributeError:
                best_iteration = None
        if isinstance(best_iteration, np.generic):
            best_iteration = int(best_iteration)
        elif isinstance(best_iteration, (float, int)):
            best_iteration = int(best_iteration)

        result = LightGBMRunResult(
            train_auroc=train_auroc,
            val_auroc=val_auroc,
            test_auroc=test_auroc,
            train_accuracy=safe_accuracy(train.y, train_pred),
            val_accuracy=safe_accuracy(val.y, val_pred),
            test_accuracy=safe_accuracy(evaluation.y, test_pred),
            train_auprc=safe_auprc(train.y, train_pred),
            val_auprc=safe_auprc(val.y, val_pred),
            test_auprc=safe_auprc(evaluation.y, test_pred),
            pretrain_val_auroc=pretrain_val_auroc,
            pretrain_val_accuracy=pretrain_val_accuracy,
            pretrain_val_auprc=pretrain_val_auprc,
            stage_durations=stage_durations,
            best_iteration=best_iteration if best_iteration is not None else None,
            booster=booster,
        )
        return result
