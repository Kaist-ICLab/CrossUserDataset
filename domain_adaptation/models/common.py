"""Shared helpers for model pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


@dataclass
class ArrayDataset:
    X: np.ndarray
    y: np.ndarray

    def is_valid(self) -> bool:
        return self.X.size > 0 and len(np.unique(self.y)) >= 2


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.size == 0:
        return float("nan")
    unique = np.unique(y_true)
    if len(unique) < 2:
        return float("nan")
    try:
        hard_pred = (y_pred >= 0.5).astype(int)
        return float(accuracy_score(y_true, hard_pred))
    except ValueError:
        return float("nan")


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return float("nan")
    if y_score.size == 0:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")
