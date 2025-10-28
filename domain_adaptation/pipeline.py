"""High-level orchestration for pretrain → fine-tune experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from .cache_utils import CacheManager
from .data_utils import align_feature_intersection, load_datasets, loso_splits, stratified_group_kfold_splits
from .models.common import ArrayDataset
from .models.tree import LightGBMConfig, LightGBMPipeline
from .models.transformer import TransformerConfig, TransformerPipeline


@dataclass
class ExperimentScenario:
    target: str
    sources: List[str]

    @property
    def name(self) -> str:
        return f"{'+'.join(self.sources)}→{self.target}"


@dataclass
class TargetSplits:
    train_features: pd.DataFrame
    val_features: pd.DataFrame
    test_features: pd.DataFrame
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray
@dataclass
class ScenarioSplit:
    split: TargetSplits
    metadata: Dict[str, object]


def _make_stratified_shuffle_split(bundle: DatasetBundle, *, seed: int, val_size: float, test_size: float) -> TargetSplits:
    X = bundle.features.values
    y = bundle.labels.astype(int)

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss_test.split(X, y))

    train_features = bundle.features.iloc[train_idx]
    train_labels = y[train_idx]
    test_features = bundle.features.iloc[test_idx]
    test_labels = y[test_idx]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_sub_idx, val_idx = next(sss_val.split(train_features.values, train_labels))

    final_train_features = train_features.iloc[train_sub_idx].reset_index(drop=True)
    final_train_labels = train_labels[train_sub_idx]
    val_features = train_features.iloc[val_idx].reset_index(drop=True)
    val_labels = train_labels[val_idx]
    test_features = test_features.reset_index(drop=True)

    return TargetSplits(
        train_features=final_train_features,
        val_features=val_features,
        test_features=test_features,
        train_labels=final_train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
    )

def _fit_scaler(
    source_frames: Sequence[pd.DataFrame],
    extra_frames: Sequence[pd.DataFrame],
) -> StandardScaler:
    scaler = StandardScaler()
    matrices: List[np.ndarray] = []
    for frame in source_frames:
        if not frame.empty:
            matrices.append(frame.values)
    for frame in extra_frames:
        if not frame.empty:
            matrices.append(frame.values)
    if not matrices:
        raise ValueError("Unable to fit scaler without any data")
    combined = np.vstack(matrices)
    scaler.fit(combined)
    return scaler


def _to_array_dataset(frame: pd.DataFrame, labels: np.ndarray, scaler: StandardScaler) -> ArrayDataset:
    if frame.empty:
        return ArrayDataset(np.empty((0, scaler.mean_.shape[0]), dtype=np.float32), np.empty((0,), dtype=np.float32))
    matrix = scaler.transform(frame.values).astype(np.float32)
    return ArrayDataset(matrix, labels.astype(np.float32))


def _combine_sources(datasets: Sequence[ArrayDataset]) -> ArrayDataset:
    if not datasets:
        return ArrayDataset(np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32))
    X = np.vstack([ds.X for ds in datasets]) if len(datasets) > 1 else datasets[0].X
    y = np.concatenate([ds.y for ds in datasets]) if len(datasets) > 1 else datasets[0].y
    return ArrayDataset(X, y)


def _split_array_dataset(
    dataset: ArrayDataset,
    val_ratio: float,
    seed: int,
) -> Tuple[ArrayDataset, Optional[ArrayDataset]]:
    if dataset.X.size == 0 or val_ratio <= 0.0:
        return dataset, None
    ratio = max(0.0, min(float(val_ratio), 0.5 if val_ratio < 1.0 else 1.0))
    stratify = dataset.y if len(np.unique(dataset.y)) > 1 else None
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            dataset.X,
            dataset.y,
            test_size=ratio,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return dataset, None
    return ArrayDataset(X_train, y_train), ArrayDataset(X_val, y_val)


def _resolve_target_ratios(
    train_ratio: Optional[float],
    val_ratio: Optional[float],
    test_ratio: Optional[float],
    fallback_val_size: float,
    fallback_test_size: float,
) -> Tuple[float, float, float]:
    test = fallback_test_size if test_ratio is None else float(test_ratio)
    if not 0 <= test < 1:
        raise ValueError("target test ratio must be in [0, 1)")
    remaining = 1.0 - test
    if remaining <= 0:
        raise ValueError("target test ratio leaves no room for train/val splits")

    if train_ratio is not None and val_ratio is not None:
        train = float(train_ratio)
        val = float(val_ratio)
    elif train_ratio is not None:
        train = float(train_ratio)
        val = remaining - train
    elif val_ratio is not None:
        val = float(val_ratio)
        train = remaining - val
    else:
        default_val = remaining * fallback_val_size
        val = default_val
        train = remaining - val

    for name, value in (("train", train), ("val", val), ("test", test)):
        if value < 0:
            raise ValueError(f"target {name} ratio became negative ({value})")
    total = train + val + test
    if total <= 0:
        raise ValueError("target split ratios sum to zero")
    if not np.isclose(total, 1.0):
        train /= total
        val /= total
        test /= total
    return float(train), float(val), float(test)


def _split_target_indices(
    labels: np.ndarray,
    ft_ratio: float,
    ft_val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    total = len(labels)
    if total == 0:
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            0.0,
        )
    if total <= 1 or ft_ratio <= 0:
        eval_idx = np.arange(total, dtype=int)
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            eval_idx,
            0.0,
        )

    ratio = float(ft_ratio)
    if ratio >= 1.0:
        ratio = 0.999
    adapt_count = max(1, int(round(total * ratio)))
    if adapt_count >= total:
        adapt_count = total - 1
    if adapt_count <= 0:
        eval_idx = np.arange(total, dtype=int)
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            eval_idx,
            0.0,
        )

    adapt_ratio = adapt_count / total
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(total - adapt_count) / total,
        random_state=seed,
    )
    dummy = np.zeros(total)
    adapt_idx, eval_idx = next(sss.split(dummy, labels))
    if adapt_idx.size == 0 or eval_idx.size == 0:
        eval_idx = np.arange(total, dtype=int)
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            eval_idx,
            0.0,
        )

    val_ratio = max(0.0, min(float(ft_val_ratio), 0.5))
    if val_ratio > 0 and adapt_idx.size > 1 and len(np.unique(labels[adapt_idx])) > 1:
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=seed,
        )
        dummy_adapt = np.zeros(adapt_idx.size)
        train_rel, val_rel = next(sss_val.split(dummy_adapt, labels[adapt_idx]))
        train_idx = adapt_idx[train_rel]
        val_idx = adapt_idx[val_rel]
    else:
        train_idx = adapt_idx
        val_idx = np.zeros((0,), dtype=int)

    actual_ratio = (train_idx.size + val_idx.size) / total
    return train_idx, val_idx, eval_idx, float(actual_ratio)


def _prepare_target_datasets(
    target_features: pd.DataFrame,
    target_labels: np.ndarray,
    scaler: StandardScaler,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> Tuple[ArrayDataset, ArrayDataset, ArrayDataset, Dict[str, int]]:
    train_frame = target_features.iloc[train_idx].reset_index(drop=True) if train_idx.size else pd.DataFrame(columns=target_features.columns)
    val_frame = target_features.iloc[val_idx].reset_index(drop=True) if val_idx.size else pd.DataFrame(columns=target_features.columns)
    eval_frame = target_features.iloc[eval_idx].reset_index(drop=True) if eval_idx.size else pd.DataFrame(columns=target_features.columns)

    train_dataset = _to_array_dataset(train_frame, target_labels[train_idx] if train_idx.size else np.empty((0,), dtype=target_labels.dtype), scaler)
    val_dataset = _to_array_dataset(val_frame, target_labels[val_idx] if val_idx.size else np.empty((0,), dtype=target_labels.dtype), scaler)
    eval_dataset = _to_array_dataset(eval_frame, target_labels[eval_idx] if eval_idx.size else np.empty((0,), dtype=target_labels.dtype), scaler)

    counts = {
        "adapt_train": int(train_idx.size),
        "adapt_val": int(val_idx.size),
        "eval": int(eval_idx.size),
    }
    return train_dataset, val_dataset, eval_dataset, counts


def _indices_to_target_split(
    bundle: DatasetBundle,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> TargetSplits:
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    train_features = bundle.features.iloc[train_idx].reset_index(drop=True)
    val_features = bundle.features.iloc[val_idx].reset_index(drop=True)
    test_features = bundle.features.iloc[test_idx].reset_index(drop=True)

    labels = bundle.labels.astype(int)

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    return TargetSplits(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
    )


def _make_groupwise_validation_split(
    train_idx: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=int)
    train_labels = labels[train_idx]
    train_groups = groups[train_idx]
    unique_groups = np.unique(train_groups)
    if unique_groups.size < 2:
        raise ValueError("Validation split requires at least two distinct groups")

    n_splits = int(min(5, unique_groups.size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_features = np.zeros_like(train_labels)
    inner_train_idx, val_idx = next(splitter.split(dummy_features, train_labels, train_groups))
    return train_idx[inner_train_idx], train_idx[val_idx]


def _generate_scenario_splits(
    bundle: DatasetBundle,
    *,
    strategy: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    group_folds: int,
) -> List[ScenarioSplit]:
    labels = bundle.labels.astype(int)
    groups = np.asarray(bundle.groups)

    if strategy == "stratified_shuffle":
        split = _make_stratified_shuffle_split(
            bundle,
            seed=seed,
            val_size=val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0.0,
            test_size=test_ratio,
        )
        return [ScenarioSplit(split=split, metadata={"split_strategy": strategy, "fold_id": 0, "test_groups": None})]

    if strategy == "loso":
        loso_pairs = loso_splits(groups)
        scenario_splits: List[ScenarioSplit] = []
        for fold_id, (train_idx, test_idx) in enumerate(loso_pairs):
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            train_final, val_idx = _make_groupwise_validation_split(train_idx, labels, groups, seed + fold_id)
            split = _indices_to_target_split(bundle, train_final, val_idx, test_idx)
            test_groups = "+".join(str(g) for g in np.unique(groups[test_idx]))
            metadata = {
                "split_strategy": strategy,
                "fold_id": fold_id,
                "test_groups": test_groups,
            }
            scenario_splits.append(ScenarioSplit(split=split, metadata=metadata))
        if not scenario_splits:
            raise ValueError("LOSO split requested but no valid folds were generated")
        return scenario_splits

    if strategy == "stratified_group_kfold":
        if group_folds < 2:
            raise ValueError("group_folds must be at least 2 for stratified_group_kfold strategy")
        fold_pairs = stratified_group_kfold_splits(
            labels,
            groups,
            n_splits=group_folds,
            shuffle=True,
            random_state=seed,
        )
        scenario_splits: List[ScenarioSplit] = []
        for fold_id, (train_idx, test_idx) in enumerate(fold_pairs):
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            train_final, val_idx = _make_groupwise_validation_split(train_idx, labels, groups, seed + fold_id)
            split = _indices_to_target_split(bundle, train_final, val_idx, test_idx)
            test_groups = "+".join(str(g) for g in np.unique(groups[test_idx]))
            metadata = {
                "split_strategy": strategy,
                "fold_id": fold_id,
                "test_groups": test_groups,
            }
            scenario_splits.append(ScenarioSplit(split=split, metadata=metadata))
        if not scenario_splits:
            raise ValueError("StratifiedGroupKFold split failed to generate folds")
        return scenario_splits

    raise ValueError(f"Unsupported split strategy '{strategy}'")


def run_experiment_scenario(
    scenario: ExperimentScenario,
    *,
    cache: CacheManager,
    overrides: Optional[Dict[str, str]] = None,
    fine_tune_ratios: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
    fine_tune_val_ratio: float = 0.2,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    model_types: Sequence[str] = ("tree", "transformer"),
    val_size: float = 0.2,
    test_size: float = 0.2,
    pretrain_val_ratio: float = 0.1,
    target_train_ratio: Optional[float] = None,
    target_val_ratio: Optional[float] = None,
    target_test_ratio: Optional[float] = None,
    split_strategy: str = "stratified_shuffle",
    group_folds: int = 5,
    lightgbm_config: Optional[LightGBMConfig] = None,
    transformer_config: Optional[TransformerConfig] = None,
) -> List[Dict[str, object]]:
    dataset_names = scenario.sources + [scenario.target]
    bundles = load_datasets(dataset_names, cache, overrides=overrides)

    aligned_bundles, feature_list = align_feature_intersection(bundles)

    source_bundles = aligned_bundles[:-1]
    target_bundle = aligned_bundles[-1]

    split_seed = seeds[0] if seeds else 42
    train_ratio, val_ratio, test_ratio = _resolve_target_ratios(
        target_train_ratio,
        target_val_ratio,
        target_test_ratio,
        val_size,
        test_size,
    )
    scenario_splits = _generate_scenario_splits(
        target_bundle,
        strategy=split_strategy,
        seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        group_folds=group_folds,
    )

    results: List[Dict[str, object]] = []

    if lightgbm_config is None:
        lightgbm_config = LightGBMConfig()
    if transformer_config is None:
        transformer_config = TransformerConfig(input_dim=len(feature_list))

    for split_entry in scenario_splits:
        split = split_entry.split
        split_meta = dict(split_entry.metadata)
        split_meta.setdefault("split_seed", split_seed)

        source_frames = [bundle.features.reset_index(drop=True) for bundle in source_bundles]
        scaler = _fit_scaler(source_frames, [])

        source_datasets = [
            _to_array_dataset(frame, bundle.labels.astype(np.float32), scaler)
            for frame, bundle in zip(source_frames, source_bundles)
        ]
        combined_pretrain = _combine_sources(source_datasets)
        pretrain_train_dataset, pretrain_val_dataset = _split_array_dataset(
            combined_pretrain,
            pretrain_val_ratio,
            seed=split_seed,
        )

        if split_meta.get("split_strategy") == "stratified_shuffle":
            target_frame_full = target_bundle.features.reset_index(drop=True)
            target_labels_full = target_bundle.labels.astype(int)
        else:
            target_frame_full = split.test_features.reset_index(drop=True)
            target_labels_full = split.test_labels.astype(int)

        total_target_samples = int(target_labels_full.size)

        for ratio in fine_tune_ratios:
            for seed in seeds:
                train_idx, val_idx, eval_idx, effective_ratio = _split_target_indices(
                    target_labels_full,
                    ratio,
                    fine_tune_val_ratio,
                    seed,
                )

                train_dataset, val_dataset, eval_dataset, target_counts = _prepare_target_datasets(
                    target_frame_full,
                    target_labels_full,
                    scaler,
                    train_idx,
                    val_idx,
                    eval_idx,
                )

                common_payload = {
                    "scenario": scenario.name,
                    "target": scenario.target,
                    "sources": "+".join(scenario.sources),
                    "feature_count": len(feature_list),
                    "ft_ratio": ratio,
                    "seed": seed,
                    "split_strategy": split_meta.get("split_strategy"),
                    "fold_id": int(split_meta.get("fold_id", 0)),
                    "test_groups": split_meta.get("test_groups"),
                    "split_seed": split_meta.get("split_seed"),
                    "train_samples": float(len(train_dataset.y)),
                    "val_samples": float(len(val_dataset.y)),
                    "pretrain_samples": float(len(combined_pretrain.y)),
                    "pretrain_train_samples": float(len(pretrain_train_dataset.y)) if pretrain_train_dataset.X.size else 0.0,
                    "pretrain_val_samples": float(len(pretrain_val_dataset.y)) if pretrain_val_dataset is not None else 0.0,
                    "adapt_samples": float(target_counts["adapt_train"] + target_counts["adapt_val"]),
                    "adapt_train_samples": float(target_counts["adapt_train"]),
                    "adapt_val_samples": float(target_counts["adapt_val"]),
                    "eval_samples": float(target_counts["eval"]),
                    "target_total_samples": float(total_target_samples),
                    "ft_ratio_effective": effective_ratio,
                }

            if "tree" in model_types:
                lgbm_pipeline = LightGBMPipeline(lightgbm_config)
                result = lgbm_pipeline.run(
                    pretrain=pretrain_train_dataset,
                    pretrain_val=pretrain_val_dataset,
                    train=train_dataset,
                    val=val_dataset,
                    adapt=None,
                    evaluation=eval_dataset,
                )
                results.append({
                    **common_payload,
                    "model": "lightgbm",
                    "mode": "pretrain_finetune",
                    "train_auroc": result.train_auroc,
                    "val_auroc": result.val_auroc,
                    "test_auroc": result.test_auroc,
                    "train_accuracy": result.train_accuracy,
                    "val_accuracy": result.val_accuracy,
                    "test_accuracy": result.test_accuracy,
                    "train_auprc": result.train_auprc,
                    "val_auprc": result.val_auprc,
                    "test_auprc": result.test_auprc,
                    "pretrain_seconds": result.stage_durations.get("pretrain_seconds"),
                    "finetune_seconds": result.stage_durations.get("finetune_seconds"),
                    "adapt_seconds": result.stage_durations.get("adapt_seconds"),
                    "pretrain_epochs": None,
                    "finetune_epochs": None,
                    "adapt_epochs": None,
                    "pretrain_val_auroc": result.pretrain_val_auroc,
                    "pretrain_val_accuracy": result.pretrain_val_accuracy,
                    "pretrain_val_auprc": result.pretrain_val_auprc,
                    "best_iteration": result.best_iteration,
                })

                if train_dataset.X.size > 0:
                    baseline_config = replace(lightgbm_config, pretrain_rounds=0)
                    baseline_pipeline = LightGBMPipeline(baseline_config)
                    baseline_result = baseline_pipeline.run(
                        pretrain=None,
                        pretrain_val=None,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "lightgbm",
                        "mode": "target_only",
                        "train_auroc": baseline_result.train_auroc,
                        "val_auroc": baseline_result.val_auroc,
                        "test_auroc": baseline_result.test_auroc,
                        "train_accuracy": baseline_result.train_accuracy,
                        "val_accuracy": baseline_result.val_accuracy,
                        "test_accuracy": baseline_result.test_accuracy,
                        "train_auprc": baseline_result.train_auprc,
                        "val_auprc": baseline_result.val_auprc,
                        "test_auprc": baseline_result.test_auprc,
                        "pretrain_seconds": baseline_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": baseline_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": baseline_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": None,
                        "finetune_epochs": None,
                        "adapt_epochs": None,
                        "pretrain_val_auroc": baseline_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": baseline_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": baseline_result.pretrain_val_auprc,
                        "best_iteration": baseline_result.best_iteration,
                    })

            if "transformer" in model_types:
                transformer_cfg = replace(transformer_config, input_dim=len(feature_list))
                transformer_pipeline = TransformerPipeline(transformer_cfg)
                transformer_result = transformer_pipeline.run(
                    seed=seed,
                    pretrain=pretrain_train_dataset,
                    pretrain_val=pretrain_val_dataset,
                    train=train_dataset,
                    val=val_dataset,
                    adapt=None,
                    evaluation=eval_dataset,
                )
                results.append({
                    **common_payload,
                    "model": "transformer",
                    "mode": "pretrain_finetune",
                    "train_auroc": transformer_result.train_auroc,
                    "val_auroc": transformer_result.val_auroc,
                    "test_auroc": transformer_result.test_auroc,
                    "train_accuracy": transformer_result.train_accuracy,
                    "val_accuracy": transformer_result.val_accuracy,
                    "test_accuracy": transformer_result.test_accuracy,
                    "train_auprc": transformer_result.train_auprc,
                    "val_auprc": transformer_result.val_auprc,
                    "test_auprc": transformer_result.test_auprc,
                    "pretrain_seconds": transformer_result.stage_durations.get("pretrain_seconds"),
                    "finetune_seconds": transformer_result.stage_durations.get("finetune_seconds"),
                    "adapt_seconds": transformer_result.stage_durations.get("adapt_seconds"),
                    "pretrain_epochs": transformer_result.stage_epochs.get("pretrain_epochs"),
                    "finetune_epochs": transformer_result.stage_epochs.get("finetune_epochs"),
                    "adapt_epochs": transformer_result.stage_epochs.get("adapt_epochs"),
                    "pretrain_val_auroc": transformer_result.pretrain_val_auroc,
                    "pretrain_val_accuracy": transformer_result.pretrain_val_accuracy,
                    "pretrain_val_auprc": transformer_result.pretrain_val_auprc,
                    "best_iteration": None,
                })

                if train_dataset.X.size > 0:
                    baseline_cfg = replace(transformer_config, input_dim=len(feature_list))
                    baseline_pipeline = TransformerPipeline(baseline_cfg)
                    baseline_result = baseline_pipeline.run(
                        seed=seed,
                        pretrain=None,
                        pretrain_val=None,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "transformer",
                        "mode": "target_only",
                        "train_auroc": baseline_result.train_auroc,
                        "val_auroc": baseline_result.val_auroc,
                        "test_auroc": baseline_result.test_auroc,
                        "train_accuracy": baseline_result.train_accuracy,
                        "val_accuracy": baseline_result.val_accuracy,
                        "test_accuracy": baseline_result.test_accuracy,
                        "train_auprc": baseline_result.train_auprc,
                        "val_auprc": baseline_result.val_auprc,
                        "test_auprc": baseline_result.test_auprc,
                        "pretrain_seconds": baseline_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": baseline_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": baseline_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": baseline_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": baseline_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": baseline_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": baseline_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": baseline_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": baseline_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

    return results


def scenarios_default() -> List[ExperimentScenario]:
    return [
        ExperimentScenario(target="D-2", sources=["D-1", "D-3"]),
        ExperimentScenario(target="D-3", sources=["D-1", "D-2"]),
        ExperimentScenario(target="D-1", sources=["D-2", "D-3"]),
    ]


def run_all_scenarios(
    *,
    cache_dir: Path,
    overrides: Optional[Dict[str, str]] = None,
    fine_tune_ratios: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
    fine_tune_val_ratio: float = 0.2,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    model_types: Sequence[str] = ("tree", "transformer"),
    val_size: float = 0.2,
    test_size: float = 0.2,
    pretrain_val_ratio: float = 0.1,
    target_train_ratio: Optional[float] = None,
    target_val_ratio: Optional[float] = None,
    target_test_ratio: Optional[float] = None,
    split_strategy: str = "stratified_shuffle",
    group_folds: int = 5,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    cache = CacheManager(cache_dir)
    all_results: List[Dict[str, object]] = []

    for scenario in scenarios_default():
        scenario_results = run_experiment_scenario(
            scenario,
            cache=cache,
            overrides=overrides,
            fine_tune_ratios=fine_tune_ratios,
            fine_tune_val_ratio=fine_tune_val_ratio,
            seeds=seeds,
            model_types=model_types,
            val_size=val_size,
            test_size=test_size,
            pretrain_val_ratio=pretrain_val_ratio,
            target_train_ratio=target_train_ratio,
            target_val_ratio=target_val_ratio,
            target_test_ratio=target_test_ratio,
            split_strategy=split_strategy,
            group_folds=group_folds,
        )
        all_results.extend(scenario_results)

    df = pd.DataFrame(all_results)
    if not df.empty:
        float_cols = df.select_dtypes(include=[np.floating]).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].round(3)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df
