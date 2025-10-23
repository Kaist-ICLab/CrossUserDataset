#!/usr/bin/env python3
"""
Cross-user model evaluation with OTDD distance correlation analysis.
Usage: python cross_user_evaluation.py [train_user] [distance_matrix_path] [model_dataset_tag]
"""

import sys
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from utility import load_selected_dataset, load_user_model_bundle


def _load_baseline_metrics(dataset_tag: str, results_dir: str = 'selected_users_dataset/results') -> pd.DataFrame:
    """Load per-user validation metrics for the dataset tag (if available)."""
    results_root = Path(results_dir)
    candidates = []

    if dataset_tag:
        tag_lower = dataset_tag.lower()
        if 'reduced' in tag_lower:
            candidates.append(results_root / 'reduced' / 'performance_features.csv')
        if 'full' in tag_lower:
            candidates.append(results_root / 'performance_features.csv')

    # Ensure default locations are also considered
    candidates.extend([
        results_root / 'performance_features.csv',
        results_root / 'reduced' / 'performance_features.csv'
    ])

    seen = set()
    unique_candidates = []
    for path in candidates:
        if path not in seen:
            unique_candidates.append(path)
            seen.add(path)

    for path in unique_candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if 'user' in df.columns:
                    return df.set_index('user')
            except Exception as exc:
                print(f"Warning: Failed to load baseline metrics from {path}: {exc}")

    print("Warning: No performance_features.csv found for the requested dataset tag.")
    return pd.DataFrame()

# Set threading for maximum performance
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

def evaluate_cross_user_performance(
    train_user: str = "P124",
    distance_matrix_path: str = None,
    model_dataset_tag: str = 'reduced_49features_normalized'
):
    """
    Train a model on train_user and evaluate on all other users.
    Correlate performance with OTDD distances.

    Args:
        train_user: User ID to train the model on
        distance_matrix_path: Path to OTDD distance matrix .npy file
    """

    print(f'=== OTDD-BASED CROSS-USER MODEL EVALUATION ===')
    print(f'Training user: {train_user}')

    # Load dataset
    print('Loading normalized dataset...')
    X, y, users, timestamps, feature_names = load_selected_dataset('reduced')
    print(f'Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(users))} users')

    user_ids = np.unique(users)
    if train_user not in user_ids:
        print(f"Error: {train_user} not found in dataset")
        return

    baseline_metrics = _load_baseline_metrics(model_dataset_tag)

    # Load distance matrix - NO FALLBACK MECHANISMS
    if distance_matrix_path and os.path.exists(distance_matrix_path):
        print(f'Loading OTDD distances from {distance_matrix_path}')
        distance_matrix = np.load(distance_matrix_path)
    elif distance_matrix_path:
        raise FileNotFoundError(f"Distance matrix file not found: {distance_matrix_path}")
    else:
        # Try default path
        default_path = f'weighted_distances_{train_user}.npy'
        if os.path.exists(default_path):
            print(f'Loading OTDD distances from {default_path}')
            distance_matrix = np.load(default_path)
        else:
            raise FileNotFoundError(f"No distance matrix found. Expected: {default_path}")

    # Get train user index and distances
    train_idx = np.where(user_ids == train_user)[0][0]
    distances_from_train = distance_matrix[train_idx, :]

    print(f'\\nOTDD-based similarity ranking for {train_user}:')
    ranking_df = pd.DataFrame({
        'user': user_ids,
        'otdd_distance': distances_from_train
    }).sort_values('otdd_distance').reset_index(drop=True)

    for i, row in ranking_df.iterrows():
        if row['user'] != train_user:
            print(f'{i:2d}. {row["user"]} (OTDD: {row["otdd_distance"]:.6f})')

    # Prepare training data
    train_mask = users == train_user
    X_train = X[train_mask]
    y_train = y[train_mask]

    print(f'\\n{train_user} training data: {len(X_train)} samples, label distribution: {np.bincount(y_train.astype(int))}')
    print(f'Loading pre-trained model for {train_user} (dataset tag: {model_dataset_tag})...')

    try:
        model_bundle = load_user_model_bundle(train_user, dataset_tag=model_dataset_tag)
    except FileNotFoundError as exc:
        print(f'Error: {exc}')
        return

    print(f"Loaded {len(model_bundle.boosters)} fold models from {model_bundle.model_dir}")

    avg_results = model_bundle.metadata.get('avg_results') if model_bundle.metadata else None
    best_cv_score = avg_results.get('val_auroc') if avg_results else None

    train_predictions = model_bundle.predict(X_train, feature_names=feature_names)
    train_auroc = roc_auc_score(y_train, train_predictions)
    train_prauc = average_precision_score(y_train, train_predictions)
    train_acc = accuracy_score(y_train, train_predictions > 0.5)

    print(f"Training set performance → AUROC={train_auroc:.4f}, PRAUC={train_prauc:.4f}, ACC={train_acc:.4f}")
    if avg_results:
        cv_auroc = avg_results.get('val_auroc')
        cv_prauc = avg_results.get('val_prauc')
        cv_acc = avg_results.get('val_acc')
        def _fmt(value):
            return float(value) if value is not None else float('nan')
        print(
            "Stored CV metrics → AUROC={:.4f}, PRAUC={:.4f}, ACC={:.4f}".format(
                _fmt(cv_auroc), _fmt(cv_prauc), _fmt(cv_acc)
            )
        )

    # Evaluate on all other users
    print('\\n=== CROSS-USER EVALUATION ===')
    results = []

    for target_user in user_ids:
        if target_user == train_user:
            continue

        target_mask = users == target_user
        X_target = X[target_mask]
        y_target = y[target_mask]

        if len(np.unique(y_target)) < 2:
            print(f'{target_user}: Skipped (only one class)')
            continue

        # Predict on target user using pre-trained model bundle
        y_pred = model_bundle.predict(X_target, feature_names=feature_names)

        # Predict and evaluate - NO TRY/EXCEPT FALLBACKS
        auroc = roc_auc_score(y_target, y_pred)
        prauc = average_precision_score(y_target, y_pred)
        accuracy = accuracy_score(y_target, y_pred > 0.5)

        # Get OTDD distance
        target_idx = np.where(user_ids == target_user)[0][0]
        otdd_dist = distance_matrix[train_idx, target_idx]

        baseline_auroc = np.nan
        baseline_prauc = np.nan
        if not baseline_metrics.empty and target_user in baseline_metrics.index:
            baseline_row = baseline_metrics.loc[target_user]
            baseline_auroc = baseline_row.get('val_auroc', np.nan)
            baseline_prauc = baseline_row.get('val_prauc', np.nan)

        delta_auroc = baseline_auroc - auroc if not np.isnan(baseline_auroc) else np.nan
        delta_prauc = baseline_prauc - prauc if not np.isnan(baseline_prauc) else np.nan

        results.append({
            'target_user': target_user,
            'otdd_distance': otdd_dist,
            'auroc': auroc,
            'prauc': prauc,
            'accuracy': accuracy,
            'baseline_auroc': baseline_auroc,
            'baseline_prauc': baseline_prauc,
            'delta_auroc': delta_auroc,
            'delta_prauc': delta_prauc,
            'samples': len(X_target),
            'label_dist': f'{np.bincount(y_target.astype(int))}'
        })

        print(f'{target_user}: AUROC={auroc:.4f}, PRAUC={prauc:.4f}, ACC={accuracy:.4f}, OTDD={otdd_dist:.6f}')

    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(results).sort_values('otdd_distance')

    print('\\n=== COMPREHENSIVE RESULTS (Sorted by OTDD Distance) ===')
    print(results_df.round(4).to_string(index=False))

    # Performance vs Distance Analysis
    if len(results_df) > 1:
        print('\\n=== PERFORMANCE VS OTDD DISTANCE ANALYSIS ===')
        auroc_corr = results_df["otdd_distance"].corr(results_df["auroc"])
        prauc_corr = results_df["otdd_distance"].corr(results_df["prauc"])
        acc_corr = results_df["otdd_distance"].corr(results_df["accuracy"])

        print(f'Correlation between OTDD distance and AUROC: {auroc_corr:.4f}')
        print(f'Correlation between OTDD distance and PRAUC: {prauc_corr:.4f}')
        print(f'Correlation between OTDD distance and Accuracy: {acc_corr:.4f}')

        if 'delta_auroc' in results_df.columns and results_df['delta_auroc'].notna().sum() > 1:
            delta_auroc_corr = results_df['otdd_distance'].corr(results_df['delta_auroc'])
            print(f'Correlation between OTDD distance and ΔAUROC (self - transfer): {delta_auroc_corr:.4f}')

        if 'delta_prauc' in results_df.columns and results_df['delta_prauc'].notna().sum() > 1:
            delta_prauc_corr = results_df['otdd_distance'].corr(results_df['delta_prauc'])
            print(f'Correlation between OTDD distance and ΔPRAUC (self - transfer): {delta_prauc_corr:.4f}')

        print('\\nTop 5 most similar users (by OTDD):')
        top5 = results_df.head(5)
        for _, row in top5.iterrows():
            print(f'{row["target_user"]}: OTDD={row["otdd_distance"]:.6f} → AUROC={row["auroc"]:.4f}')

        print('\\nTop 5 least similar users (by OTDD):')
        bottom5 = results_df.tail(5)
        for _, row in bottom5.iterrows():
            print(f'{row["target_user"]}: OTDD={row["otdd_distance"]:.6f} → AUROC={row["auroc"]:.4f}')

    # Save results
    output_file = f'{train_user}_cross_user_evaluation.csv'
    results_df.to_csv(output_file, index=False)
    print(f'\\nResults saved to {output_file}')

    # Summary statistics
    if len(results_df) > 0:
        print('\\n=== SUMMARY STATISTICS ===')
        if best_cv_score is not None:
            print(f'Training user {train_user} stored CV AUROC: {best_cv_score:.4f}')
        else:
            print(f'Training user {train_user} training AUROC (computed): {train_auroc:.4f}')
        print(f'Cross-user AUROC - Mean: {results_df["auroc"].mean():.4f}, Std: {results_df["auroc"].std():.4f}')
        print(f'Cross-user PRAUC - Mean: {results_df["prauc"].mean():.4f}, Std: {results_df["prauc"].std():.4f}')
        print(f'Cross-user Accuracy - Mean: {results_df["accuracy"].mean():.4f}, Std: {results_df["accuracy"].std():.4f}')

        # Best transferring users
        best_auroc = results_df.nlargest(3, 'auroc')
        print('\\nBest transferring users (highest cross-user AUROC):')
        for _, row in best_auroc.iterrows():
            print(f'{row["target_user"]}: AUROC={row["auroc"]:.4f}, OTDD={row["otdd_distance"]:.6f}')

    return results_df

if __name__ == "__main__":
    # Parse command line arguments
    train_user = sys.argv[1] if len(sys.argv) > 1 else "P124"
    distance_matrix_path = sys.argv[2] if len(sys.argv) > 2 else None
    model_dataset_tag = sys.argv[3] if len(sys.argv) > 3 else 'reduced_49features_normalized'

    print(f"Cross-user evaluation for training user: {train_user}")
    if distance_matrix_path:
        print(f"Using distance matrix: {distance_matrix_path}")
    print(f"Using trained model dataset tag: {model_dataset_tag}")

    evaluate_cross_user_performance(train_user, distance_matrix_path, model_dataset_tag)
