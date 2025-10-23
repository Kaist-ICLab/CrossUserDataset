#!/usr/bin/env python3
"""
Create reduced feature dataset using union of important features
Usage: python reduce_features.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.training.train_models import train_user_models

DATASET_DIR = ROOT_DIR / 'selected_users_dataset'
RESULTS_DIR = DATASET_DIR / 'results'
MODELS_DIR = DATASET_DIR / 'models'

def create_reduced_dataset():
    # Load feature importance results
    feature_summary_path = RESULTS_DIR / 'feature_summary.csv'
    if not feature_summary_path.exists():
        raise FileNotFoundError(f"Feature summary not found at {feature_summary_path}. Run training first.")

    feature_summary = pd.read_csv(feature_summary_path)

    # Ensure feature names are accessible (older exports keep them in an unnamed index column)
    first_col = feature_summary.columns[0]
    if first_col.startswith('Unnamed') or first_col == '':
        feature_summary = feature_summary.rename(columns={first_col: 'feature'})

    # Identify feature name column
    feature_col_candidates = [
        col for col in feature_summary.columns
        if col.lower().startswith('feature')
    ]
    feature_col = feature_col_candidates[0] if feature_col_candidates else feature_summary.columns[0]

    # Determine which column contains aggregated importance
    if 'mean_importance' in feature_summary.columns:
        importance_col = 'mean_importance'
    elif 'mean' in feature_summary.columns:
        importance_col = 'mean'
    else:
        raise KeyError(
            "Expected an aggregated importance column ('mean_importance' or 'mean') in feature_summary.csv"
        )

    # Get features with 1%+ importance (union across all users)
    important_features = (
        feature_summary[feature_summary[importance_col] >= 1.0][feature_col]
        .astype(str)
        .tolist()
    )

    if not important_features:
        raise ValueError(
            "No features met the 1% importance threshold. Verify feature_summary.csv contents."
        )

    # Load full dataset
    full_dataset_path = DATASET_DIR / 'full_216features.pkl'
    with open(full_dataset_path, 'rb') as f:
        X, y, users, timestamps, feature_names = pickle.load(f)

    # Ensure X is a numpy array and feature_names aligns with its columns
    if hasattr(X, 'values'):
        X_df = X
        X = X_df.values
        if feature_names is None or (hasattr(feature_names, '__len__') and len(feature_names) != X.shape[1]):
            feature_names = list(X_df.columns)
    else:
        X = np.asarray(X)

    # Extract feature indices with fallback for descriptive feature names
    feature_indices = []
    for feat in important_features:
        if isinstance(feat, str) and feat.startswith('feature_'):
            feature_indices.append(int(feat.split('_')[1]))
        elif feat in feature_names:
            feature_indices.append(feature_names.index(feat))
        else:
            raise ValueError(f"Unable to parse feature index from '{feat}'.")

    # Determine mapping back to original feature indices
    mapping_path = DATASET_DIR / 'feature_mapping.pkl'
    mapping = {}
    base_original_indices = None
    if mapping_path.exists():
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        candidates = mapping.get('full_original_indices') or mapping.get('original_indices')
        if isinstance(candidates, dict):
            candidates = list(candidates.values())
        if candidates is not None:
            base_original_indices = list(candidates)

    if base_original_indices is None:
        base_original_indices = list(range(len(feature_names)))

    if feature_indices and max(feature_indices) >= len(base_original_indices):
        base_original_indices = list(range(len(feature_names)))

    # Optionally load pre-normalized full dataset to avoid redundant work
    normalized_full_path = DATASET_DIR / 'full_216features_normalized.pkl'
    X_full_normalized = None
    if normalized_full_path.exists():
        with open(normalized_full_path, 'rb') as f:
            X_norm_loaded, _, _, _, feature_names_norm = pickle.load(f)

        if hasattr(X_norm_loaded, 'values'):
            X_norm_array = X_norm_loaded.values
            feature_names_norm = list(X_norm_loaded.columns)
        else:
            X_norm_array = np.asarray(X_norm_loaded)

        if len(feature_names_norm) != X_norm_array.shape[1]:
            raise ValueError(
                "Normalized full dataset feature_names length does not match array width"
            )

        feature_names_norm = [str(name) for name in feature_names_norm]
        feature_names_aligned = [str(name) for name in feature_names]
        if feature_names_norm != feature_names_aligned:
            name_to_idx = {name: idx for idx, name in enumerate(feature_names_norm)}
            try:
                reorder_indices = [name_to_idx[name] for name in feature_names_aligned]
            except KeyError as exc:
                raise KeyError(
                    f"Feature {exc} missing in normalized full dataset; please regenerate datasets."
                ) from exc
            X_norm_array = X_norm_array[:, reorder_indices]

        X_full_normalized = X_norm_array

    # Create reduced dataset
    X_reduced = X[:, feature_indices]
    feature_names_reduced = [str(feature_names[i]) for i in feature_indices]

    # Save reduced dataset
    reduced_dataset_path = DATASET_DIR / 'reduced_49features.pkl'
    with open(reduced_dataset_path, 'wb') as f:
        pickle.dump((X_reduced, y, users, timestamps, feature_names_reduced), f)

    # Save normalized reduced dataset if pre-normalized data exists
    precomputed_normalized = None
    if X_full_normalized is not None:
        precomputed_normalized = X_full_normalized[:, feature_indices]

    if precomputed_normalized is not None:
        create_normalized_dataset(
            X_reduced,
            y,
            users,
            timestamps,
            feature_names_reduced,
            precomputed=precomputed_normalized
        )

    # Update mapping
    mapping_reduced = {
        'original_indices': [int(base_original_indices[i]) for i in feature_indices],
        'feature_indices': [int(i) for i in feature_indices],
        'original_feature_count': len(base_original_indices),
        'reduced_feature_count': len(feature_indices),
        'reduction_percentage': (
            (1 - len(feature_indices)/len(base_original_indices)) * 100
            if base_original_indices else 0.0
        ),
        'selected_features': feature_names_reduced,
        'full_original_indices': base_original_indices
    }

    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping_reduced, f)

    print(f"Reduced dataset: {X_reduced.shape[1]} features ({mapping_reduced['reduction_percentage']:.1f}% reduction)")
    return X_reduced, y, users, timestamps, feature_names_reduced, feature_indices


def create_normalized_dataset(
    X,
    y,
    users,
    timestamps,
    feature_names,
    precomputed: np.ndarray = None
):
    """Create and persist a user-wise normalized reduced dataset."""
    normalized_path = DATASET_DIR / 'reduced_49features_normalized.pkl'

    if precomputed is not None:
        X_normalized = precomputed
    else:
        from sklearn.preprocessing import StandardScaler

        X_normalized = np.zeros_like(X, dtype=float)
        for user in np.unique(users):
            user_mask = users == user
            scaler = StandardScaler()
            X_normalized[user_mask] = scaler.fit_transform(X[user_mask])

    with open(normalized_path, 'wb') as f:
        pickle.dump((X_normalized, y, users, timestamps, feature_names), f)

    print(f"Normalized reduced dataset saved to {normalized_path}")
    return normalized_path


def recompute_feature_importances(normalized_dataset_path):
    """Re-train per-user models on the reduced dataset to refresh feature importances."""
    print("\nRecomputing per-user models and feature importances on the reduced dataset...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_user_models(
        dataset_path=str(normalized_dataset_path),
        results_dir=str(RESULTS_DIR),
        results_subdir='reduced',
        model_output_dir=str(MODELS_DIR),
        save_models=True,
        apply_scaler=False
    )

if __name__ == "__main__":
    X_reduced, y, users, timestamps, feature_names, feature_indices = create_reduced_dataset()
    normalized_path = DATASET_DIR / 'reduced_49features_normalized.pkl'
    # if not normalized_path.exists():
    #     normalized_path = create_normalized_dataset(X_reduced, y, users, timestamps, feature_names)
    recompute_feature_importances(normalized_path)
