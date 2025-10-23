#!/usr/bin/env python3
"""
Compare performance between full and reduced feature datasets
Usage: python compare_performance.py
"""

import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

# Set threading to use ALL cores but prevent process-level conflicts
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())  # Use ALL cores for OpenMP threading
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())  # Use ALL cores for BLAS
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())  # Use ALL cores for MKL
os.environ['NUMBA_NUM_THREADS'] = str(os.cpu_count())  # Use ALL cores for Numba
# Prevent multiprocessing to avoid core competition
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def objective(trial, X_train, y_train, X_val, y_val):
    # Check if validation set has both classes
    if len(np.unique(y_val)) < 2:
        return 0.5  # Return baseline score if only one class

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'num_threads': os.cpu_count(),  # Use ALL cores for threading
        'verbosity': -1
    }

    # Create clean feature names for LightGBM
    clean_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=clean_names)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(params, train_data, valid_sets=[val_data],
                     callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)])

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)

    try:
        return roc_auc_score(y_val, y_pred)
    except ValueError:
        return 0.5  # Return baseline if ROC AUC calculation fails

def train_user_models(dataset_path, normalize_per_user=True):
    with open(dataset_path, 'rb') as f:
        X, y, users, timestamps, feature_names = pickle.load(f)

    # Clean feature names for LightGBM compatibility
    clean_feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    results = []

    for user in np.unique(users):
        print(f"Training model for {user}... (normalize_per_user={normalize_per_user})")

        user_mask = users == user
        X_user = X[user_mask]
        y_user = y[user_mask]

        if normalize_per_user:
            scaler = StandardScaler()
            X_user_scaled = scaler.fit_transform(X_user)
        else:
            X_user_scaled = X_user

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_user_scaled, y_user)):
            # Handle both numpy array and DataFrame indexing
            if isinstance(X_user_scaled, np.ndarray):
                X_train, X_test = X_user_scaled[train_idx], X_user_scaled[test_idx]
            else:
                X_train, X_test = X_user_scaled.iloc[train_idx], X_user_scaled.iloc[test_idx]
            y_train, y_test = y_user[train_idx], y_user[test_idx]

            # Skip if test set doesn't have both classes
            if len(np.unique(y_test)) < 2:
                print(f"  Fold {fold}: Skipping due to single class in test set")
                continue

            # For validation split, use stratified split to ensure both classes
            if len(y_train) > 10:
                try:
                    from sklearn.model_selection import train_test_split
                    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
                        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
                    )
                    # Double check validation set has both classes
                    if len(np.unique(y_val_opt)) < 2:
                        X_val_opt, y_val_opt = X_test, y_test  # Use test set as validation
                except:
                    val_split = int(0.8 * len(X_train))
                    if isinstance(X_train, np.ndarray):
                        X_train_opt, X_val_opt = X_train[:val_split], X_train[val_split:]
                    else:
                        X_train_opt, X_val_opt = X_train.iloc[:val_split], X_train.iloc[val_split:]
                    y_train_opt, y_val_opt = y_train[:val_split], y_train[val_split:]
            else:
                val_split = int(0.8 * len(X_train))
                if isinstance(X_train, np.ndarray):
                    X_train_opt, X_val_opt = X_train[:val_split], X_train[val_split:]
                else:
                    X_train_opt, X_val_opt = X_train.iloc[:val_split], X_train.iloc[val_split:]
                y_train_opt, y_val_opt = y_train[:val_split], y_train[val_split:]

            # Create study with pruning - PURE SEQUENTIAL to maximize threading efficiency
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
            # Sequential optimization - each LightGBM trial uses ALL cores via threading
            study.optimize(
                lambda trial: objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt),
                n_trials=50,  # Back to 50 since each trial is much faster with all cores
                show_progress_bar=False
                # No n_jobs parameter = pure sequential, no process spawning
            )

            best_params = study.best_params
            best_params.update({
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_threads': os.cpu_count(),  # Use ALL cores for threading
                'verbosity': -1
            })

            # Use clean feature names for final model too
            clean_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=clean_names)
            model = lgb.train(best_params, train_data, num_boost_round=1000)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Safe metric calculation with error handling
            try:
                train_auroc = roc_auc_score(y_train, y_train_pred)
                train_prauc = average_precision_score(y_train, y_train_pred)
                train_acc = accuracy_score(y_train, y_train_pred > 0.5)

                test_auroc = roc_auc_score(y_test, y_test_pred)
                test_prauc = average_precision_score(y_test, y_test_pred)
                test_acc = accuracy_score(y_test, y_test_pred > 0.5)

                fold_results.append({
                    'fold': fold,
                    'train_auroc': train_auroc, 'train_prauc': train_prauc, 'train_acc': train_acc,
                    'test_auroc': test_auroc, 'test_prauc': test_prauc, 'test_acc': test_acc
                })
            except ValueError as e:
                print(f"  Fold {fold}: Skipping due to metric calculation error: {e}")
                continue

        # Only calculate averages if we have valid fold results
        if len(fold_results) > 0:
            avg_results = {
                'user': user,
                'train_auroc': np.mean([r['train_auroc'] for r in fold_results]),
                'train_prauc': np.mean([r['train_prauc'] for r in fold_results]),
                'train_acc': np.mean([r['train_acc'] for r in fold_results]),
                'val_auroc': np.mean([r['test_auroc'] for r in fold_results]),
                'val_prauc': np.mean([r['test_prauc'] for r in fold_results]),
                'val_acc': np.mean([r['test_acc'] for r in fold_results])
            }
            results.append(avg_results)
        else:
            print(f"  User {user}: No valid folds, skipping user")

    return pd.DataFrame(results), None

def compare_datasets():
    print("=== COMPARING DATASETS WITH DIFFERENT NORMALIZATION ===\n")

    # Test with user-wise normalization
    print("1. Training models on full 216-feature dataset (user-wise normalized)...")
    results_full_norm, _ = train_user_models('../../selected_users_dataset/full_216features.pkl', normalize_per_user=True)

    print("\n2. Training models on reduced 49-feature dataset (user-wise normalized)...")
    results_reduced_norm, _ = train_user_models('../../selected_users_dataset/reduced_49features.pkl', normalize_per_user=True)

    # Test without normalization
    print("\n3. Training models on full 216-feature dataset (no normalization)...")
    results_full_raw, _ = train_user_models('../../selected_users_dataset/full_216features.pkl', normalize_per_user=False)

    print("\n4. Training models on reduced 49-feature dataset (no normalization)...")
    results_reduced_raw, _ = train_user_models('../../selected_users_dataset/reduced_49features.pkl', normalize_per_user=False)

    # Compare normalized versions
    print("\n=== NORMALIZED COMPARISON (User-wise StandardScaler) ===")
    comparison_norm = results_full_norm.merge(results_reduced_norm, on='user', suffixes=('_full', '_reduced'))

    for metric in ['auroc', 'prauc', 'acc']:
        for split in ['train', 'val']:
            col_full = f'{split}_{metric}_full'
            col_reduced = f'{split}_{metric}_reduced'
            retention_col = f'{split}_{metric}_retention'
            comparison_norm[retention_col] = (comparison_norm[col_reduced] / comparison_norm[col_full] * 100).round(2)

    print(f"Features: 216 → 49 ({((216-49)/216*100):.1f}% reduction)")
    for split in ['train', 'val']:
        for metric in ['auroc', 'prauc', 'acc']:
            full_mean = comparison_norm[f'{split}_{metric}_full'].mean()
            reduced_mean = comparison_norm[f'{split}_{metric}_reduced'].mean()
            retention_mean = comparison_norm[f'{split}_{metric}_retention'].mean()
            print(f"{split.upper()} {metric.upper()}: {full_mean:.3f} → {reduced_mean:.3f} ({retention_mean:.1f}% retention)")

    # Compare raw versions
    print("\n=== RAW COMPARISON (No normalization) ===")
    comparison_raw = results_full_raw.merge(results_reduced_raw, on='user', suffixes=('_full', '_reduced'))

    for metric in ['auroc', 'prauc', 'acc']:
        for split in ['train', 'val']:
            col_full = f'{split}_{metric}_full'
            col_reduced = f'{split}_{metric}_reduced'
            retention_col = f'{split}_{metric}_retention'
            comparison_raw[retention_col] = (comparison_raw[col_reduced] / comparison_raw[col_full] * 100).round(2)

    print(f"Features: 216 → 49 ({((216-49)/216*100):.1f}% reduction)")
    for split in ['train', 'val']:
        for metric in ['auroc', 'prauc', 'acc']:
            full_mean = comparison_raw[f'{split}_{metric}_full'].mean()
            reduced_mean = comparison_raw[f'{split}_{metric}_reduced'].mean()
            retention_mean = comparison_raw[f'{split}_{metric}_retention'].mean()
            print(f"{split.upper()} {metric.upper()}: {full_mean:.3f} → {reduced_mean:.3f} ({retention_mean:.1f}% retention)")

    # Normalization impact analysis
    print("\n=== NORMALIZATION IMPACT ANALYSIS ===")

    # Full dataset: normalized vs raw
    print("Full 216-feature dataset:")
    for split in ['train', 'val']:
        for metric in ['auroc', 'prauc', 'acc']:
            norm_mean = results_full_norm[f'{split}_{metric}'].mean()
            raw_mean = results_full_raw[f'{split}_{metric}'].mean()
            improvement = ((norm_mean - raw_mean) / raw_mean * 100)
            print(f"  {split.upper()} {metric.upper()}: Raw {raw_mean:.3f} → Normalized {norm_mean:.3f} ({improvement:+.1f}%)")

    # Reduced dataset: normalized vs raw
    print("Reduced 49-feature dataset:")
    for split in ['train', 'val']:
        for metric in ['auroc', 'prauc', 'acc']:
            norm_mean = results_reduced_norm[f'{split}_{metric}'].mean()
            raw_mean = results_reduced_raw[f'{split}_{metric}'].mean()
            improvement = ((norm_mean - raw_mean) / raw_mean * 100)
            print(f"  {split.upper()} {metric.upper()}: Raw {raw_mean:.3f} → Normalized {norm_mean:.3f} ({improvement:+.1f}%)")

    # Save all results - create directory if needed
    import os
    results_dir = '../../selected_users_dataset/results'
    os.makedirs(results_dir, exist_ok=True)

    comparison_norm.to_csv(f'{results_dir}/performance_comparison_normalized.csv', index=False)
    comparison_raw.to_csv(f'{results_dir}/performance_comparison_raw.csv', index=False)

    # Top performers comparison
    print(f"\nTop performers (val AUROC, normalized):")
    top_users_norm = comparison_norm.nlargest(5, 'val_auroc_reduced')[['user', 'val_auroc_full', 'val_auroc_reduced', 'val_auroc_retention']]
    for _, row in top_users_norm.iterrows():
        print(f"{row['user']}: {row['val_auroc_full']:.3f} → {row['val_auroc_reduced']:.3f} ({row['val_auroc_retention']:.1f}%)")

    print(f"\nTop performers (val AUROC, raw):")
    top_users_raw = comparison_raw.nlargest(5, 'val_auroc_reduced')[['user', 'val_auroc_full', 'val_auroc_reduced', 'val_auroc_retention']]
    for _, row in top_users_raw.iterrows():
        print(f"{row['user']}: {row['val_auroc_full']:.3f} → {row['val_auroc_reduced']:.3f} ({row['val_auroc_retention']:.1f}%)")

    return {
        'normalized': comparison_norm,
        'raw': comparison_raw,
        'full_norm': results_full_norm,
        'full_raw': results_full_raw,
        'reduced_norm': results_reduced_norm,
        'reduced_raw': results_reduced_raw
    }

if __name__ == "__main__":
    compare_datasets()