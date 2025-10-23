#!/usr/bin/env python3
"""
Enhanced utility functions for stress detection with weighted OTDD distance calculation.
Uses feature importance weights from training users to compute distance metrics.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from otdd.pytorch.distance import DatasetDistance
from joblib import Parallel, delayed


def load_feature_importance_weights(
    user: str,
    feature_list: List[str],
    results_dir: str = "selected_users_dataset/results",
    results_subdir: str = None
) -> np.ndarray:
    """
    Load feature importance weights for a specific user, matching the provided feature list.
    Works with matching dimensions only (49-feature importance for 49-feature data, 216 for 216).

    Args:
        user (str): User ID (e.g., 'P124')
        feature_list (List[str]): List of feature names in current dataset
        results_dir (str): Directory containing feature importance CSV files

    Returns:
        np.ndarray: Normalized feature importance weights matching feature_list order
    """
    results_path = Path(results_dir)
    candidate_paths = []

    if results_subdir:
        candidate_paths.append(results_path / results_subdir / f"{user}_feature_importance.csv")
    else:
        feature_count = len(feature_list)
        if feature_count and feature_count <= 64:  # heuristic for reduced feature sets
            candidate_paths.append(results_path / 'reduced' / f"{user}_feature_importance.csv")
        candidate_paths.append(results_path / f"{user}_feature_importance.csv")

    importance_df = None
    for path in candidate_paths:
        if path.exists():
            importance_df = pd.read_csv(path)
            break

    if importance_df is None:
        print(f"Warning: Feature importance file not found for {user}, using uniform weights")
        return np.ones(len(feature_list)) / len(feature_list)

    if 'feature_name' in importance_df.columns:
        feature_column = 'feature_name'
    elif 'feature' in importance_df.columns:
        feature_column = 'feature'
    else:
        feature_column = importance_df.columns[0]

    if 'normalized_importance' in importance_df.columns:
        importance_values = importance_df['normalized_importance'].to_numpy(dtype=float)
    else:
        importance_values = importance_df['importance'].to_numpy(dtype=float)

    feature_to_importance = dict(zip(importance_df[feature_column], importance_values))
    default_value = float(importance_values.mean()) if len(importance_values) else 1.0

    weights = np.array([
        feature_to_importance.get(feature_name, default_value)
        for feature_name in feature_list
    ], dtype=float)

    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)

    return weights


class UserModelBundle:
    """Container for persisted per-user LightGBM models and preprocessing."""

    def __init__(
        self,
        user: str,
        boosters: List[lgb.Booster],
        scaler: Optional[Any],
        feature_names: List[str],
        metadata: Dict[str, Any],
        model_dir: Path
    ) -> None:
        self.user = user
        self.boosters = boosters
        self.scaler = scaler
        self.feature_names = feature_names
        self.metadata = metadata
        self.model_dir = model_dir

    def _ensure_feature_order(self, X: Any, feature_names: Optional[List[str]]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            missing = set(self.feature_names) - set(feature_names)
            if missing:
                raise ValueError(f"Missing features for prediction: {missing}")
            return X[self.feature_names].to_numpy(dtype=float)

        array = np.asarray(X, dtype=float)
        if feature_names and list(feature_names) != list(self.feature_names):
            index_map = []
            name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
            for name in self.feature_names:
                if name not in name_to_idx:
                    raise ValueError(f"Missing feature '{name}' in provided data")
                index_map.append(name_to_idx[name])
            array = array[:, index_map]
        return array

    def transform(self, X: Any, feature_names: Optional[List[str]] = None) -> np.ndarray:
        data = self._ensure_feature_order(X, feature_names)
        if self.scaler is not None:
            data = self.scaler.transform(data)
        return data

    def predict(self, X: Any, feature_names: Optional[List[str]] = None) -> np.ndarray:
        data = self.transform(X, feature_names)
        if not self.boosters:
            raise RuntimeError("No trained boosters available for user model bundle")
        predictions = np.vstack([booster.predict(data) for booster in self.boosters])
        return predictions.mean(axis=0)


def load_user_model_bundle(
    user: str,
    dataset_tag: str = 'reduced_49features_normalized',
    model_root: str = 'selected_users_dataset/models'
) -> UserModelBundle:
    """Load persisted LightGBM boosters and preprocessing artifacts for a user."""

    model_root_path = Path(model_root)
    if not model_root_path.exists():
        raise FileNotFoundError(f"Model root directory not found: {model_root_path}")

    user_dir = model_root_path / dataset_tag / user

    if not user_dir.exists():
        # Fallback: search for user directory under any dataset tag
        for candidate in model_root_path.iterdir():
            potential_dir = candidate / user
            if potential_dir.exists():
                user_dir = potential_dir
                break

    info_path = user_dir / 'model_info.json'
    if not info_path.exists():
        raise FileNotFoundError(f"Model metadata not found for {user} at {info_path}")

    with open(info_path, 'r') as f:
        metadata = json.load(f)

    feature_names = metadata.get('feature_names', [])
    boosters = []
    for model_file in metadata.get('fold_model_files', []):
        model_path = user_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file for {user}: {model_path}")
        boosters.append(lgb.Booster(model_file=str(model_path)))

    scaler = None
    scaler_file = metadata.get('scaler_file')
    if scaler_file:
        scaler_path = user_dir / scaler_file
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

    return UserModelBundle(user, boosters, scaler, feature_names, metadata, user_dir)


def calculate_weighted_otdd_distance(
    features_u: np.ndarray,
    labels_u: np.ndarray,
    features_v: np.ndarray,
    labels_v: np.ndarray,
    importances_u: np.ndarray,
    importances_v: np.ndarray,
    ot_params: Dict[str, Any]
) -> float:
    """
    Calculate the OTDD distance between two user datasets with weighted features.
    NO FALLBACK - if OTDD fails, the function will raise an exception.

    Args:
        features_u: Feature matrix for user U.
        labels_u: Label array for user U.
        features_v: Feature matrix for user V.
        labels_v: Label array for user V.
        importances_u: Feature importances for user U.
        importances_v: Feature importances for user V.
        ot_params: Parameters for the DatasetDistance calculation.

    Returns:
        OTDD distance as a float.

    Raises:
        Exception: If OTDD calculation fails for any reason.
    """
    # Subsample if datasets are too large
    max_samples = 200
    if len(features_u) > max_samples:
        np.random.seed(42)
        idx_u = np.random.choice(len(features_u), max_samples, replace=False)
        features_u = features_u[idx_u]
        labels_u = labels_u[idx_u]
        print(f"Subsampled user U from {len(features_u)} to {max_samples} samples")

    if len(features_v) > max_samples:
        np.random.seed(42)
        idx_v = np.random.choice(len(features_v), max_samples, replace=False)
        features_v = features_v[idx_v]
        labels_v = labels_v[idx_v]
        print(f"Subsampled user V from {len(features_v)} to {max_samples} samples")

    # Features should already be normalized if using normalized dataset

    weights = importances_u
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)

    # Clip weights to avoid extreme scaling and ensure no zeros
    weights = np.clip(weights, 1e-4, 1.0)  # Minimum weight increased
    weights = weights / weights.sum()  # Renormalize after clipping

    scale = np.sqrt(weights)[None, :]
    Xu = (features_u * scale).astype(np.float32)
    Xv = (features_v * scale).astype(np.float32)

    # Add small noise to avoid identical samples
    Xu += np.random.normal(0, 1e-3, Xu.shape).astype(np.float32)
    Xv += np.random.normal(0, 1e-3, Xv.shape).astype(np.float32)

    # Check for NaN/inf values
    if not (np.isfinite(Xu).all() and np.isfinite(Xv).all()):
        raise ValueError("NaN or inf values detected in scaled features")

    yu = labels_u.astype(np.int64)
    yv = labels_v.astype(np.int64)

    ds_u = TensorDataset(torch.from_numpy(Xu), torch.from_numpy(yu))
    ds_v = TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))

    # Use smaller batch sizes
    batch_size_u = min(32, len(Xu))
    batch_size_v = min(32, len(Xv))

    loader_u = DataLoader(ds_u, batch_size=batch_size_u, shuffle=False)
    loader_v = DataLoader(ds_v, batch_size=batch_size_v, shuffle=False)

    ot = DatasetDistance(loader_u, loader_v, **ot_params)
    distance = ot.distance().item()

    return distance


def _compute_otdd_for_pair(
    i: int,
    j: int,
    user_ids: np.ndarray,
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    feature_list: List[str],
    user_importances: Dict[str, np.ndarray],
    ot_params: Dict[str, Any]
) -> Tuple[int, int, float]:
    """
    Compute OTDD distance for a single pair of users.

    Args:
        i: Index of first user in user_ids.
        j: Index of second user in user_ids.
        user_ids: Array of unique user identifiers.
        data: Mapping from user ID to (features, labels).
        feature_list: List of features to include.
        user_importances: Mapping from user ID to feature importances array.
        ot_params: Parameters for the DatasetDistance calculation.

    Returns:
        Tuple of (i, j, distance).
    """
    u_id = user_ids[i]
    v_id = user_ids[j]
    Xu, yu = data[u_id]
    Xv, yv = data[v_id]
    imp_u = user_importances.get(u_id, np.ones(len(feature_list)) / len(feature_list))
    imp_v = user_importances.get(v_id, np.ones(len(feature_list)) / len(feature_list))
    dist = calculate_weighted_otdd_distance(
        Xu, yu, Xv, yv, imp_u, imp_v, ot_params
    )
    return i, j, dist


def compute_pairwise_otdd_matrix(
    features: pd.DataFrame,
    labels: np.ndarray,
    group_indices: np.ndarray,
    feature_list: List[str],
    user_importances: Dict[str, np.ndarray],
    device: str = "cpu",
    n_jobs: int = -1,
    cache_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise OTDD distances between users and return user IDs with the distance matrix.
    Results are cached to disk to avoid recomputation.

    Args:
        features: Dataframe of all user features.
        labels: Array of all labels corresponding to features.
        group_indices: Array of user identifiers for each row in features.
        feature_list: List of features to include.
        user_importances: Mapping from user ID to feature importances array.
        device: PyTorch device for OTDD computation.
        n_jobs: Number of parallel jobs for distance computation.
        cache_path: File path for caching the distance matrix.

    Returns:
        Tuple containing array of unique user IDs and the distance matrix.
    """
    user_ids = np.unique(group_indices)
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached OTDD distance matrix from {cache_path}")
        return user_ids, np.load(cache_path)

    data = {
        u: (
            features.loc[group_indices == u, feature_list].values,
            labels[group_indices == u]
        )
        for u in user_ids
    }

    ot_params = {
        "inner_ot_method": "exact",
        "debiased_loss": True,
        "p": 2,
        "λ_x": 1.0,
        "λ_y": 1.0,
        "entreg": 1e-2,
        "device": device
    }

    pairs = [(i, j) for i in range(len(user_ids)) for j in range(i + 1, len(user_ids))]

    print(f"Computing {len(pairs)} pairwise OTDD distances...")
    if n_jobs == 1:
        # Sequential computation
        results = []
        for idx, (i, j) in enumerate(pairs):
            print(f"OTDD {idx+1}/{len(pairs)}: {user_ids[i]} vs {user_ids[j]}")
            result = _compute_otdd_for_pair(
                i, j, user_ids, data, feature_list, user_importances, ot_params
            )
            results.append(result)
    else:
        # Parallel computation
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_compute_otdd_for_pair)(
                i, j, user_ids, data, feature_list, user_importances, ot_params
            )
            for i, j in pairs
        )

    U = len(user_ids)
    D = np.zeros((U, U), dtype=float)
    for i, j, dist in results:
        D[i, j] = D[j, i] = dist

    if np.isnan(D).any():
        print("Warning: Found NaN values in OTDD distance matrix")
        max_val = np.nanmax(D)
        D[np.isnan(D)] = max_val * 10

    if cache_path:
        print(f"Saving OTDD distance matrix to {cache_path}")
        np.save(cache_path, D)

    return user_ids, D


def calculate_user_similarity_ranking(
    train_user: str,
    features: pd.DataFrame,
    labels: np.ndarray,
    group_indices: np.ndarray,
    feature_list: List[str],
    results_dir: str = "selected_users_dataset/results",
    results_subdir: str = None,
    cache_path: str = None
) -> pd.DataFrame:
    """
    Calculate similarity ranking of all users relative to a training user
    using OTDD distance with feature importance weights.

    Args:
        train_user: Reference user ID for weighting and ranking
        features: DataFrame of all user features
        labels: Array of all labels
        group_indices: Array of user identifiers
        feature_list: List of features to include
        results_dir: Directory with feature importance files
        cache_path: Optional cache file path

    Returns:
        DataFrame with users ranked by similarity to training user
    """
    # Load feature importance weights for all users
    user_ids = np.unique(group_indices)
    user_importances = {}
    for u in user_ids:
        weights = load_feature_importance_weights(u, feature_list, results_dir, results_subdir)
        user_importances[u] = weights

    print(f"Computing OTDD distances with feature importance weights...")
    print(f"Reference user {train_user} importance weights loaded: {user_importances[train_user] is not None}")

    # Compute OTDD distance matrix
    user_ids, distance_matrix = compute_pairwise_otdd_matrix(
        features, labels, group_indices, feature_list,
        user_importances, device="cpu", n_jobs=1, cache_path=cache_path
    )

    # Find training user index
    train_idx = np.where(user_ids == train_user)[0][0]

    # Get distances from training user to all others
    distances_from_train = distance_matrix[train_idx, :]

    # Create ranking DataFrame
    ranking_df = pd.DataFrame({
        'user': user_ids,
        'distance_from_train': distances_from_train,
        'similarity_rank': np.argsort(distances_from_train) + 1
    })

    # Sort by similarity (lowest distance = highest similarity)
    ranking_df = ranking_df.sort_values('distance_from_train').reset_index(drop=True)
    ranking_df['similarity_rank'] = range(1, len(ranking_df) + 1)

    print(f"\nOTDD similarity ranking relative to {train_user}:")
    print(f"Most similar users to {train_user}:")
    for i, row in ranking_df.iterrows():
        if row['user'] != train_user:
            print(f"{row['similarity_rank']:2d}. {row['user']} (OTDD distance: {row['distance_from_train']:.4f})")

    return ranking_df


# Load datasets convenience function
def load_selected_dataset(dataset_type: str = "full") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load selected users dataset.

    Args:
        dataset_type: "full" for 216 features or "reduced" for 49 features

    Returns:
        Tuple of (X, y, users, timestamps, feature_names)
    """
    if dataset_type == "full":
        filepath = "selected_users_dataset/full_216features_normalized.pkl"
    elif dataset_type == "reduced":
        filepath = "selected_users_dataset/reduced_49features_normalized.pkl"
    elif dataset_type in {"full_leaf", "leaf_full"}:
        filepath = "selected_users_dataset/full_216features_leaf.pkl"
    elif dataset_type in {"reduced_leaf", "leaf_reduced"}:
        filepath = "selected_users_dataset/reduced_49features_leaf.pkl"
        print('yes')
    else:
        raise ValueError("dataset_type must be 'full', 'reduced', 'full_leaf', or 'reduced_leaf'")

    with open(filepath, 'rb') as f:
        X, y, users, timestamps, feature_names = pickle.load(f)

    # Convert DataFrame to numpy array if needed
    if hasattr(X, 'values'):
        print("Converting DataFrame to numpy array...")
        feature_names = list(X.columns)
        X = X.values

    # Fix feature names if they're incorrect (should match X.shape[1])
    if len(feature_names) != X.shape[1]:
        print(f"Warning: Feature names length ({len(feature_names)}) doesn't match X.shape[1] ({X.shape[1]})")
        print("Generating correct feature names...")
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    return X, y, users, timestamps, feature_names


if __name__ == "__main__":
    # Example usage
    print("Loading selected users dataset...")
    X, y, users, timestamps, feature_names = load_selected_dataset("reduced_leaf")

    print(f"Dataset shape: {X.shape}")
    print(f"Users: {np.unique(users)}")

    # Example: Calculate similarity ranking relative to P124
    features_df = pd.DataFrame(X, columns=feature_names)

    print("\nCalculating weighted OTDD similarity ranking...")
    ranking = calculate_user_similarity_ranking(
        train_user="P052",
        features=features_df,
        labels=y,
        group_indices=users,
        feature_list=feature_names,
        cache_path="otdd_distances_P052.npy"
    )

    print(f"\nComplete ranking saved. Total users: {len(ranking)}")
