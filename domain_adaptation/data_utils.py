"""Dataset loading and feature-selection helpers for domain adaptation experiments."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .cache_utils import CacheManager


DEFAULT_DATASET_PATHS: Mapping[str, str] = {
    # Guesses based on the developer environment. Override as needed.
    "D-1": "~/Desktop/KAIST-collab/UBICOMP/data/D-2/stress_binary_personal-full.pkl",
    "D-2": "~/Desktop/KAIST-collab/UBICOMP/data/D-3/stress_binary_personal-full_D#3.pkl",
    "D-3": "~/Desktop/KAIST-collab/UBICOMP/data/D-4/stress_binary_personal-current.pkl",
}

FEATURE_NAME_NORMALIZATION: Mapping[str, str] = {
    "BAT_PLG#VAL=UNDEFINED": "BAT_PLG#VAL=UNKNOWN",
    "CALL_CNT#VAL=기타": "CALL_CNT#VAL=OTHER",
    "CALL_CNT#VAL=휴대전화": "CALL_CNT#VAL=MOBILE",
    "CALL_CNT#VAL=휴대폰": "CALL_CNT#VAL=MOBILE",
}


@dataclass
class DatasetConfig:
    name: str
    path: Path

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> Dict[str, "DatasetConfig"]:
        configs: Dict[str, DatasetConfig] = {}
        for name, path in mapping.items():
            configs[name] = cls(name=name, path=Path(path).expanduser().resolve())
        return configs


@dataclass
class DatasetBundle:
    name: str
    features: pd.DataFrame
    labels: np.ndarray
    groups: np.ndarray
    timestamps: np.ndarray
    metadata: MutableMapping[str, object] = field(default_factory=dict)

    @property
    def feature_names(self) -> List[str]:
        return list(self.features.columns)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.features.shape

    def with_features(self, feature_frame: pd.DataFrame) -> "DatasetBundle":
        return DatasetBundle(
            name=self.name,
            features=feature_frame.copy(),
            labels=self.labels.copy(),
            groups=self.groups.copy(),
            timestamps=self.timestamps.copy(),
            metadata=dict(self.metadata),
        )

    def select_features(self, feature_names: Sequence[str]) -> "DatasetBundle":
        aligned = self.features.reindex(columns=list(feature_names), fill_value=0.0)
        return self.with_features(aligned)

    def to_matrix(self, feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        if feature_names is None:
            return self.features.values.astype(np.float32)
        aligned = self.features.reindex(columns=list(feature_names), fill_value=0.0)
        return aligned.values.astype(np.float32)


def _ensure_dataframe(features_obj, feature_names: Optional[Sequence[str]]) -> pd.DataFrame:
    if isinstance(features_obj, pd.DataFrame):
        return features_obj.copy()
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(features_obj.shape[1])]
    return pd.DataFrame(np.asarray(features_obj), columns=list(feature_names))


def _normalize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    if not FEATURE_NAME_NORMALIZATION:
        return df
    renamed = df.rename(columns=FEATURE_NAME_NORMALIZATION)
    if renamed.columns.duplicated().any():
        renamed = renamed.T.groupby(level=0, sort=False).sum().T
    return renamed


def _extract_tuple_payload(obj: Sequence) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if len(obj) < 3:
        raise ValueError("Dataset tuple/list must contain at least X, y, and users/groups arrays")

    X = obj[0]
    y = np.asarray(obj[1])
    groups = np.asarray(obj[2])
    timestamps = np.asarray(obj[3]) if len(obj) >= 4 else np.arange(len(y))
    feature_names = None
    if len(obj) >= 5:
        maybe_names = obj[4]
        if isinstance(maybe_names, (list, tuple)) and len(maybe_names) == np.asarray(X).shape[1]:
            feature_names = list(maybe_names)
    df = _ensure_dataframe(X, feature_names)
    metadata: Dict[str, object] = {}
    if len(obj) > 5:
        metadata["extra"] = obj[5:]
    return df, y, groups, timestamps, metadata


def _extract_dict_payload(obj: Mapping) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    keys_lower = {k.lower(): k for k in obj.keys()}
    metadata: Dict[str, object] = {}
    if "x" in keys_lower and "y" in keys_lower:
        X = obj[keys_lower["x"]]
        y = np.asarray(obj[keys_lower["y"]])

        groups_key = keys_lower.get("groups") or keys_lower.get("users")
        if groups_key and groups_key in obj:
            groups = np.asarray(obj[groups_key])
        else:
            groups = np.zeros_like(y)

        ts_key = keys_lower.get("timestamps") or keys_lower.get("t") or keys_lower.get("datetimes")
        if ts_key and ts_key in obj:
            timestamps = np.asarray(obj[ts_key])
        else:
            timestamps = np.arange(len(y))

        feature_names = obj.get("feature_names")
        df = _ensure_dataframe(X, feature_names if feature_names is not None else None)

        excluded_keys = {
            keys_lower["x"],
            keys_lower["y"],
            groups_key,
            ts_key,
            "feature_names",
        }
        metadata.update({k: v for k, v in obj.items() if k not in excluded_keys and not str(k).startswith("_")})
        return df, y, groups, timestamps, metadata

    if "x_common" in keys_lower and "y" in keys_lower:
        X = obj[keys_lower["x_common"]]
        y = np.asarray(obj[keys_lower["y"]])
        groups_key = keys_lower.get("groups") or keys_lower.get("users") or "groups"
        groups = np.asarray(obj[groups_key])
        ts_key = keys_lower.get("datetimes") or keys_lower.get("timestamps") or keys_lower.get("t")
        if ts_key and ts_key in obj:
            timestamps = np.asarray(obj[ts_key])
        else:
            timestamps = np.arange(len(y))
        df = _ensure_dataframe(X, None)
        excluded_keys = set(keys_lower.values())
        metadata.update({k: v for k, v in obj.items() if k not in excluded_keys and not str(k).startswith("_")})
        return df, y, groups, timestamps, metadata

    raise ValueError("Unsupported dataset dictionary structure")


def load_dataset(path: Path, name: str) -> DatasetBundle:
    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    if isinstance(payload, dict):
        df, y, groups, timestamps, metadata = _extract_dict_payload(payload)
    elif isinstance(payload, (list, tuple)):
        df, y, groups, timestamps, metadata = _extract_tuple_payload(payload)
    else:
        raise TypeError(f"Unsupported dataset format in {path}")

    metadata.setdefault("source_path", str(path))
    normalized_df = _normalize_feature_names(df)
    return DatasetBundle(
        name=name,
        features=normalized_df,
        labels=y,
        groups=groups,
        timestamps=timestamps,
        metadata=metadata,
    )


@dataclass
class DatasetStore:
    configs: Dict[str, DatasetConfig]
    cache: CacheManager

    @classmethod
    def from_defaults(
        cls,
        cache: CacheManager,
        overrides: Optional[Mapping[str, str]] = None,
    ) -> "DatasetStore":
        mapping: Dict[str, str] = dict(DEFAULT_DATASET_PATHS)
        if overrides:
            for key, value in overrides.items():
                mapping[key] = value
        configs = DatasetConfig.from_mapping(mapping)
        return cls(configs=configs, cache=cache)

    def load(self, name: str) -> DatasetBundle:
        if name not in self.configs:
            raise KeyError(f"Unknown dataset name '{name}'. Available: {sorted(self.configs)}")
        config = self.configs[name]
        bundle = load_dataset(config.path, name)
        bundle.metadata.setdefault("dataset_name", name)
        return bundle

    def load_many(self, names: Sequence[str]) -> List[DatasetBundle]:
        return [self.load(name) for name in names]


# ---------------------------------------------------------------------------
# Feature alignment helpers
# ---------------------------------------------------------------------------


def align_feature_intersection(
    bundles: Sequence[DatasetBundle],
) -> Tuple[List[DatasetBundle], List[str]]:
    if not bundles:
        return [], []
    common: set[str] = set(bundles[0].feature_names)
    for bundle in bundles[1:]:
        common &= set(bundle.feature_names)
    if not common:
        raise ValueError("No common features across provided datasets")
    ordered = [name for name in bundles[0].feature_names if name in common]
    aligned = [bundle.select_features(ordered) for bundle in bundles]
    return aligned, ordered


def load_datasets(
    dataset_names: Sequence[str],
    cache: CacheManager,
    overrides: Optional[Mapping[str, str]] = None,
) -> List[DatasetBundle]:
    store = DatasetStore.from_defaults(cache=cache, overrides=overrides)
    return store.load_many(dataset_names)


# ---------------------------------------------------------------------------
# Splitting utilities
# ---------------------------------------------------------------------------


def loso_splits(groups: Sequence) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate Leave-One-Subject-Out splits based on group identifiers.

    Args:
        groups: Sequence of subject/user identifiers aligned with the dataset rows.

    Returns:
        List of (train_idx, test_idx) tuples covering each unique group.
    """
    groups_array = np.asarray(groups)
    unique_groups = np.unique(groups_array)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for group in unique_groups:
        test_idx = np.where(groups_array == group)[0]
        train_idx = np.where(groups_array != group)[0]
        if len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def stratified_group_kfold_splits(
    labels: Sequence,
    groups: Sequence,
    n_splits: int,
    *,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified group k-fold splits while respecting group boundaries.

    Args:
        labels: Target labels used for stratification.
        groups: Group identifiers (e.g., user IDs).
        n_splits: Number of folds.
        shuffle: Whether to shuffle group assignments before splitting.
        random_state: Optional seed for reproducibility when shuffling.

    Returns:
        List of (train_idx, test_idx) tuples.
    """
    labels_array = np.asarray(labels)
    groups_array = np.asarray(groups)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    dummy_features = np.zeros_like(labels_array)
    for train_idx, test_idx in splitter.split(dummy_features, labels_array, groups_array):
        splits.append((train_idx, test_idx))
    return splits
