#!/usr/bin/env python3
"""Generate LightGBM leaf-index embeddings for a dataset.

Given a (normalized) feature dataset and label file, this script trains (or
loads) a global LightGBM model, converts each sample into a one-hot leaf-index
embedding, and saves the resulting dataset as a pickle:
    (X_leaf, y, users, timestamps, feature_names_leaf)

Example:
    python generate_leaf_embeddings.py \
        --dataset-path selected_users_dataset/full_216features_normalized.pkl \
        --output-path selected_users_dataset/full_216features_leaf.pkl

To reuse an existing LightGBM model:
    python generate_leaf_embeddings.py \
        --dataset-path selected_users_dataset/full_216features_normalized.pkl \
        --booster-path models/global_full_leaf_model.txt
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def _load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    with open(path, 'rb') as f:
        X, y, users, timestamps, feature_names = pickle.load(f)

    if not isinstance(feature_names, list):
        feature_names = list(feature_names)
    feature_names = [str(name) for name in feature_names]
    X = np.asarray(X)
    y = np.asarray(y)
    users = np.asarray(users)
    timestamps = np.asarray(timestamps)
    return X, y, users, timestamps, feature_names


def _train_global_model(
    X: np.ndarray,
    y: np.ndarray,
    num_boost_round: int,
    num_leaves: int,
    learning_rate: float,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    min_data_in_leaf: int,
    num_threads: int,
    valid_fraction: float,
    random_state: int,
    early_stopping_rounds: int,
    eval_period: int,
) -> Tuple[lgb.Booster, Dict[str, Dict[str, float]]]:
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'min_data_in_leaf': min_data_in_leaf,
        'verbosity': -1,
        'num_threads': num_threads,
    }

    use_validation = 0.0 < valid_fraction < 1.0

    if use_validation:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=valid_fraction,
            random_state=random_state
        )
        train_idx, val_idx = next(splitter.split(X, y))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        print(f"LightGBM training set: {X_train.shape[0]} samples; validation set: {X_val.shape[0]} samples")
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        callbacks = []
        if early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        if eval_period > 0:
            callbacks.append(lgb.log_evaluation(eval_period))
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
    else:
        print("LightGBM training without explicit validation split")
        X_train, y_train = X, y
        X_val = y_val = None
        train_data = lgb.Dataset(X_train, label=y_train)
        callbacks = [lgb.log_evaluation(eval_period)] if eval_period > 0 else []
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=callbacks
        )

    def _best_iteration(model: lgb.Booster) -> int:
        best_iter = model.best_iteration
        if best_iter is None or best_iter <= 0:
            try:
                best_iter = model.current_iteration()
            except AttributeError:
                best_iter = None
        if best_iter is None or best_iter <= 0:
            best_iter = model.num_trees()
        if best_iter is None or best_iter <= 0:
            best_iter = num_boost_round
        return best_iter

    def _compute_split_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        y_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        try:
            metrics['logloss'] = log_loss(y_true, y_clipped)
        except ValueError:
            metrics['logloss'] = float('nan')
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_pred)
        except ValueError:
            metrics['auroc'] = float('nan')
        try:
            metrics['prauc'] = average_precision_score(y_true, y_pred)
        except ValueError:
            metrics['prauc'] = float('nan')
        print(
            f"{name} metrics → LogLoss={metrics['logloss']:.4f}, "
            f"AUROC={metrics['auroc']:.4f}, PRAUC={metrics['prauc']:.4f}"
        )
        return metrics

    metrics: Dict[str, Dict[str, float]] = {}
    best_iter = _best_iteration(booster)

    train_pred = booster.predict(X_train, num_iteration=best_iter)
    metrics['train'] = _compute_split_metrics('Train', y_train, train_pred)

    if use_validation and X_val is not None:
        val_pred = booster.predict(X_val, num_iteration=best_iter)
        metrics['valid'] = _compute_split_metrics('Valid', y_val, val_pred)
    else:
        print("Validation disabled; skipping validation metrics.")

    return booster, metrics


def _build_one_hot_embeddings(leaf_indices: np.ndarray, booster: lgb.Booster) -> Tuple[np.ndarray, List[str]]:
    model_dump = booster.dump_model()
    tree_infos = model_dump['tree_info']
    num_leaves_per_tree = [tree['num_leaves'] for tree in tree_infos]

    offsets = np.cumsum([0] + num_leaves_per_tree)
    total_leaves = offsets[-1]
    n_samples, n_trees = leaf_indices.shape
    embeddings = np.zeros((n_samples, total_leaves), dtype=np.float32)

    for tree_idx in range(n_trees):
        start, end = offsets[tree_idx], offsets[tree_idx + 1]
        tree_leaf_indices = leaf_indices[:, tree_idx]
        embeddings[np.arange(n_samples), start + tree_leaf_indices] = 1.0

    feature_names = []
    for tree_idx, num_leaves in enumerate(num_leaves_per_tree):
        for leaf_idx in range(num_leaves):
            feature_names.append(f'tree{tree_idx}_leaf{leaf_idx}')

    return embeddings, feature_names


def generate_leaf_embeddings(
    dataset_path: Path,
    output_path: Path,
    booster_path: Optional[Path],
    num_boost_round: int,
    num_leaves: int,
    learning_rate: float,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    min_data_in_leaf: int,
    num_threads: int,
    valid_fraction: float,
    random_state: int,
    early_stopping_rounds: int,
    eval_period: int,
    save_booster_path: Optional[Path],
) -> None:
    X, y, users, timestamps, feature_names = _load_dataset(dataset_path)
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(users))} users")

    metrics: Dict[str, Dict[str, float]]
    if booster_path is not None:
        booster = lgb.Booster(model_file=str(booster_path))
        print(f"Loaded existing LightGBM model from {booster_path}")
        probabilities = booster.predict(X)
        metrics = {
            'train': {
                'logloss': log_loss(y, np.clip(probabilities, 1e-7, 1 - 1e-7)),
                'auroc': roc_auc_score(y, probabilities),
                'prauc': average_precision_score(y, probabilities)
            }
        }
        print(
            f"Existing model metrics → LogLoss={metrics['train']['logloss']:.4f}, "
            f"AUROC={metrics['train']['auroc']:.4f}, PRAUC={metrics['train']['prauc']:.4f}"
        )
    else:
        print("Training global LightGBM model for leaf embeddings...")
        booster, metrics = _train_global_model(
            X,
            y,
            num_boost_round=num_boost_round,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            min_data_in_leaf=min_data_in_leaf,
            num_threads=num_threads,
            valid_fraction=valid_fraction,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            eval_period=eval_period,
        )
        if save_booster_path is not None:
            booster.save_model(str(save_booster_path))
            print(f"Saved trained LightGBM model to {save_booster_path}")

    print("Computing leaf indices...")
    leaf_indices = booster.predict(X, pred_leaf=True)
    print(f"Leaf indices shape: {leaf_indices.shape}")

    print("Converting to one-hot leaf embeddings...")
    embeddings, leaf_feature_names = _build_one_hot_embeddings(leaf_indices, booster)
    print(f"Leaf embedding matrix shape: {embeddings.shape}")

    output = (
        embeddings,
        y,
        users,
        timestamps,
        leaf_feature_names
    )

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"Leaf embeddings saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LightGBM leaf-index embeddings for a dataset.")
    parser.add_argument('--dataset-path', type=Path, required=True,
                        help='Input dataset pickle (preferably user-wise normalized).')
    parser.add_argument('--output-path', type=Path, required=True,
                        help='Output pickle path for the leaf embeddings.')
    parser.add_argument('--booster-path', type=Path, default=None,
                        help='Optional: pre-trained LightGBM model to use instead of training a new one.')
    parser.add_argument('--save-booster-path', type=Path, default=None,
                        help='Optional: where to save the trained LightGBM model.')
    parser.add_argument('--num-boost-round', type=int, default=500,
                        help='Number of boosting rounds when training a new LightGBM model.')
    parser.add_argument('--num-leaves', type=int, default=64,
                        help='num_leaves parameter for LightGBM training.')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='learning_rate parameter for LightGBM training.')
    parser.add_argument('--feature-fraction', type=float, default=0.8,
                        help='feature_fraction parameter for LightGBM training.')
    parser.add_argument('--bagging-fraction', type=float, default=0.8,
                        help='bagging_fraction parameter for LightGBM training.')
    parser.add_argument('--bagging-freq', type=int, default=1,
                        help='bagging_freq parameter for LightGBM training.')
    parser.add_argument('--min-data-in-leaf', type=int, default=20,
                        help='min_data_in_leaf parameter for LightGBM training.')
    parser.add_argument('--num-threads', type=int, default=-1,
                        help='LightGBM num_threads. Use -1 for all available cores.')
    parser.add_argument('--valid-fraction', type=float, default=0.2,
                        help='Fraction of data reserved for validation when training a new LightGBM model.')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for the train/validation split.')
    parser.add_argument('--early-stopping-rounds', type=int, default=50,
                        help='Early stopping patience for LightGBM (0 to disable).')
    parser.add_argument('--eval-period', type=int, default=50,
                        help='Print LightGBM evaluation metrics every N rounds (0 to disable).')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_leaf_embeddings(
        dataset_path=args.dataset_path.expanduser().resolve(),
        output_path=args.output_path.expanduser().resolve(),
        booster_path=(args.booster_path.expanduser().resolve() if args.booster_path else None),
        num_boost_round=args.num_boost_round,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        min_data_in_leaf=args.min_data_in_leaf,
        num_threads=args.num_threads,
        valid_fraction=args.valid_fraction,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        eval_period=args.eval_period,
        save_booster_path=(args.save_booster_path.expanduser().resolve() if args.save_booster_path else None),
    )
