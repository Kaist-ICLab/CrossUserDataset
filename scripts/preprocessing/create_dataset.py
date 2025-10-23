#!/usr/bin/env python3
"""Build raw and normalized datasets for selected users.

This script loads the original personal stress dataset, optionally filters
users (by explicit list and/or label imbalance), and writes both the raw
feature matrix and a user-wise normalized version into `selected_users_dataset/`.

Usage example:
    python create_dataset.py \
        --source-pkl ~/Archived/stress_binary_personal-current.pkl \
        --output-dir selected_users_dataset \
        --selected-users P024 P045 P046 \
        --max-label-ratio 3.0
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _load_dataset(source_pkl: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(source_pkl, 'rb') as f:
        loaded = pickle.load(f)

    if isinstance(loaded, dict):
        try:
            return loaded['X'], loaded['y'], loaded['users'], loaded['timestamps']
        except KeyError as exc:
            raise KeyError(f"Missing key in source pickle: {exc}") from exc

    if isinstance(loaded, (list, tuple)):
        if len(loaded) >= 4:
            return loaded[0], loaded[1], loaded[2], loaded[3]
        raise ValueError(
            f"Expected at least four elements in source pickle, found {len(loaded)}"
        )

    raise TypeError("Unsupported pickle format; expected tuple/list/dict with X, y, users, timestamps")


def _filter_by_ratio(
    users: np.ndarray,
    labels: np.ndarray,
    max_ratio: float
) -> np.ndarray:
    stats: List[Dict[str, object]] = []
    for user in np.unique(users):
        user_mask = users == user
        counts = np.bincount(labels[user_mask].astype(int), minlength=2)
        maj = counts.max()
        min_ = counts[counts > 0].min() if np.any(counts > 0) else 0
        ratio = float('inf') if min_ == 0 else maj / min_
        stats.append({'user_id': user, 'label_ratio': ratio})

    stats_df = pd.DataFrame(stats)
    allowed = stats_df[stats_df['label_ratio'] < max_ratio]['user_id']
    print("\nLabel-ratio filter summary:")
    print(stats_df.to_string(index=False))
    return np.isin(users, allowed.values)


def _normalise_per_user(
    X: np.ndarray,
    users: np.ndarray
) -> np.ndarray:
    X_normalized = np.zeros_like(X, dtype=float)
    for user in np.unique(users):
        user_mask = users == user
        scaler = StandardScaler()
        X_normalized[user_mask] = scaler.fit_transform(X[user_mask])
    return X_normalized


def _determine_selected_users(
    users: np.ndarray,
    selected_users: Optional[Sequence[str]]
) -> np.ndarray:
    if not selected_users:
        return np.ones_like(users, dtype=bool)
    selected_set = set(selected_users)
    mask = np.isin(users, list(selected_set))
    missing = selected_set - set(users[mask])
    if missing:
        print(f"Warning: requested users not found in source data: {sorted(missing)}")
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Create raw & normalized datasets for selected users.")
    parser.add_argument('--source-pkl', type=Path, required=True,
                        help='Path to the original dataset pickle (X, y, users, timestamps).')
    parser.add_argument('--output-dir', type=Path, default=Path('selected_users_dataset'),
                        help='Directory where the datasets will be written.')
    parser.add_argument('--selected-users', nargs='*', default=None,
                        help='Explicit list of user IDs to keep. If omitted, keep all users.')
    parser.add_argument('--selected-users-file', type=Path, default=None,
                        help='Optional file containing one user ID per line to keep.')
    parser.add_argument('--max-label-ratio', type=float, default=2.5,
                        help='Optional strict upper bound on majority/minority label ratio.')

    args = parser.parse_args()

    X, y, users, timestamps = _load_dataset(args.source_pkl.expanduser().resolve())
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    selected_users: List[str] = []
    if args.selected_users_file:
        with open(args.selected_users_file, 'r') as f:
            selected_users.extend([line.strip() for line in f if line.strip()])
    if args.selected_users:
        selected_users.extend(args.selected_users)

    mask = _determine_selected_users(users, selected_users if selected_users else None)

    if args.max_label_ratio is not None:
        ratio_mask = _filter_by_ratio(users, y, args.max_label_ratio)
        mask &= ratio_mask

    if not np.any(mask):
        raise ValueError("No users remain after applying filters.")

    X_selected = np.asarray(X)[mask]
    y_selected = np.asarray(y)[mask]
    users_selected = np.asarray(users)[mask]
    timestamps_selected = np.asarray(timestamps)[mask]

    X_normalized = _normalise_per_user(X_selected, users_selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / 'full_216features.pkl'
    normalized_path = args.output_dir / 'full_216features_normalized.pkl'

    with open(raw_path, 'wb') as f:
        pickle.dump((X_selected, y_selected, users_selected, timestamps_selected, feature_names), f)

    with open(normalized_path, 'wb') as f:
        pickle.dump((X_normalized, y_selected, users_selected, timestamps_selected, feature_names), f)

    print(f"Saved raw dataset to {raw_path}")
    print(f"Saved normalized dataset to {normalized_path}")
    print(f"Users retained: {sorted(np.unique(users_selected))}")


if __name__ == '__main__':
    main()
