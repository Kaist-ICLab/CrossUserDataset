#!/usr/bin/env python3
"""
Train LightGBM models for each user with optional selection and parallelism.

Usage examples:

# Train on normalized full dataset, filter users by AUROC ≥ 0.8, and save models
python scripts/training/train_models.py \
  --dataset-path selected_users_dataset/full_216features_normalized.pkl \
  --raw-dataset-path selected_users_dataset/full_216features.pkl \
  --selection-threshold 0.8 \
  --save-models \
  --user-jobs 4

# Final training only (no selection phase)
python scripts/training/train_models.py \
  --dataset-path selected_users_dataset/full_216features_normalized.pkl \
  --save-models
"""

import argparse
import concurrent.futures
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import TrialPruned
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def _resolve_model_root(dataset_path: Path, model_output_dir: Optional[os.PathLike] = None) -> Path:
    if model_output_dir is not None:
        return Path(model_output_dir)
    return dataset_path.parent / 'models'


def load_user_model_bundle(
    user: str,
    dataset_tag: Optional[str] = None,
    model_output_dir: Optional[os.PathLike] = None
) -> Dict[str, Any]:
    model_root = Path(model_output_dir) if model_output_dir else Path('selected_users_dataset/models')

    if dataset_tag is None:
        candidates = [d for d in model_root.iterdir() if d.is_dir() and (d / user).exists()]
        if not candidates:
            raise FileNotFoundError(f"No saved models found for user {user} in {model_root}")
        if len(candidates) > 1:
            print(f"Warning: multiple datasets for {user} found. Using {candidates[0].name}.")
        dataset_dir = candidates[0]
    else:
        dataset_dir = model_root / dataset_tag
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {dataset_dir}")

    user_dir = dataset_dir / user
    if not user_dir.exists():
        raise FileNotFoundError(f"Saved model directory for {user} not found: {user_dir}")

    metadata_path = user_dir / 'model_info.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found for {user}: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    fold_files: List[str] = metadata.get('fold_model_files', [])
    if not fold_files:
        raise ValueError(f"No fold models recorded in metadata for {user} ({metadata_path})")

    models: List[lgb.Booster] = []
    for model_file in fold_files:
        model_path = user_dir / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        models.append(lgb.Booster(model_file=str(model_path)))

    scaler = None
    if metadata.get('apply_scaler') and metadata.get('scaler_file'):
        scaler_path = user_dir / metadata['scaler_file']
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            print(f"Warning: scaler file referenced but not found ({scaler_path}). Proceeding without scaler.")

    metadata['model_dir'] = str(user_dir)
    metadata.setdefault('dataset_tag', dataset_dir.name)

    return {
        'models': models,
        'scaler': scaler,
        'metadata': metadata,
        'model_dir': user_dir
    }


def _safe_roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float('nan')


def _safe_average_precision(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except ValueError:
        return float('nan')


def _train_single_user(
    user: str,
    X: np.ndarray,
    y: np.ndarray,
    users: np.ndarray,
    feature_names: List[str],
    dataset_path: Path,
    dataset_tag: str,
    target_results_dir: Path,
    model_dataset_dir: Optional[Path],
    save_models: bool,
    apply_scaler: bool,
    n_trials: int,
    num_boost_round: int,
    threads_per_model: int
) -> Optional[Dict[str, Any]]:
    try:
        user_mask = users == user
        X_user = X[user_mask]
        y_user = y[user_mask].astype(int)

        unique_labels = np.unique(y_user)
        if len(unique_labels) < 2:
            print(f"Skipping {user}: dataset contains a single class ({unique_labels[0]}).")
            return None

        class_counts = np.bincount(y_user)
        class_counts = class_counts[class_counts > 0]
        min_class_samples = int(class_counts.min())

        if min_class_samples < 2:
            print(f"Skipping {user}: not enough samples in minority class (min count={min_class_samples}).")
            return None

        n_splits = min(5, min_class_samples)
        if n_splits < 2:
            print(f"Skipping {user}: insufficient data for cross-validation (n_splits={n_splits}).")
            return None

        scaler = StandardScaler() if apply_scaler else None
        X_user_scaled = scaler.fit_transform(X_user) if scaler is not None else X_user

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        fold_importances = []
        fold_best_params = []
        fold_model_files = []

        user_model_dir = None
        if save_models and model_dataset_dir is not None:
            user_model_dir = model_dataset_dir / user
            if user_model_dir.exists():
                shutil.rmtree(user_model_dir)
            user_model_dir.mkdir(parents=True, exist_ok=True)

        def objective_wrapper(trial, X_tr, y_tr, X_val, y_val):
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                raise TrialPruned("Validation split must contain at least two classes")

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
                'verbosity': -1,
                'num_threads': threads_per_model
            }

            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(params, train_data, valid_sets=[val_data],
                             callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)])

            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            return roc_auc_score(y_val, y_pred)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_user_scaled, y_user)):
            X_train, X_test = X_user_scaled[train_idx], X_user_scaled[test_idx]
            y_train, y_test = y_user[train_idx], y_user[test_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"  {user} Fold {fold}: skipped due to insufficient class diversity.")
                continue

            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=min(0.2, 0.5),
                random_state=42 + fold
            )
            try:
                inner_train_idx, inner_val_idx = next(sss.split(X_train, y_train))
            except ValueError:
                print(f"  {user} Fold {fold}: unable to create stratified validation split; skipping.")
                continue

            X_train_opt, X_val_opt = X_train[inner_train_idx], X_train[inner_val_idx]
            y_train_opt, y_val_opt = y_train[inner_train_idx], y_train[inner_val_idx]

            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: objective_wrapper(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt),
                n_trials=n_trials,
                show_progress_bar=False
            )

            try:
                best_params = study.best_params
            except ValueError:
                print(f"  {user} Fold {fold}: no successful trials; skipping.")
                continue

            best_params_with_obj = best_params.copy()
            best_params_with_obj.update({
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'num_threads': threads_per_model
            })
            fold_best_params.append(best_params_with_obj.copy())

            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(best_params_with_obj, train_data, num_boost_round=num_boost_round)

            if save_models and user_model_dir is not None:
                fold_model_path = user_model_dir / f'fold_{fold}.txt'
                model.save_model(str(fold_model_path))
                fold_model_files.append(fold_model_path.name)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_auroc = _safe_roc_auc(y_train, y_train_pred)
            train_prauc = _safe_average_precision(y_train, y_train_pred)
            train_acc = accuracy_score(y_train, y_train_pred > 0.5)

            test_auroc = _safe_roc_auc(y_test, y_test_pred)
            test_prauc = _safe_average_precision(y_test, y_test_pred)
            test_acc = accuracy_score(y_test, y_test_pred > 0.5)

            fold_results.append({
                'fold': fold,
                'train_auroc': train_auroc,
                'train_prauc': train_prauc,
                'train_acc': train_acc,
                'test_auroc': test_auroc,
                'test_prauc': test_prauc,
                'test_acc': test_acc
            })

            importance = model.feature_importance(importance_type='gain')
            fold_importances.append(importance)

        if not fold_results or not fold_importances:
            print(f"Skipping {user}: no valid folds were completed.")
            return None

        if save_models and scaler is not None and user_model_dir is not None:
            joblib.dump(scaler, user_model_dir / 'scaler.joblib')

        avg_results = {
            'user': user,
            'train_auroc': np.mean([r['train_auroc'] for r in fold_results]),
            'train_prauc': np.mean([r['train_prauc'] for r in fold_results]),
            'train_acc': np.mean([r['train_acc'] for r in fold_results]),
            'val_auroc': np.mean([r['test_auroc'] for r in fold_results]),
            'val_prauc': np.mean([r['test_prauc'] for r in fold_results]),
            'val_acc': np.mean([r['test_acc'] for r in fold_results])
        }

        if save_models and user_model_dir is not None:
            metadata = {
                'user': user,
                'dataset_path': str(dataset_path),
                'dataset_tag': dataset_tag,
                'feature_names': feature_names,
                'apply_scaler': apply_scaler,
                'scaler_file': 'scaler.joblib' if scaler is not None else None,
                'fold_model_files': fold_model_files,
                'fold_best_params': fold_best_params,
                'fold_metrics': fold_results,
                'avg_results': avg_results,
                'label_distribution': np.bincount(y_user.astype(int)).tolist(),
                'total_samples': int(len(X_user)),
                'num_trials': int(n_trials),
                'num_boost_round': int(num_boost_round)
            }
            with open(user_model_dir / 'model_info.json', 'w') as f:
                json.dump(metadata, f, indent=2)

        importance_matrix = np.vstack(fold_importances)
        avg_importance = importance_matrix.mean(axis=0)
        importance_std = importance_matrix.std(axis=0)
        total_importance = float(np.sum(avg_importance))
        normalized_importance = avg_importance / total_importance if total_importance > 0 else np.zeros_like(avg_importance)

        user_importance_df = pd.DataFrame({
            'user': user,
            'feature_index': np.arange(len(feature_names)),
            'feature_name': feature_names,
            'importance': avg_importance.astype(float),
            'importance_std': importance_std.astype(float),
            'normalized_importance': normalized_importance.astype(float)
        })
        user_importance_df = user_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        user_importance_df['rank'] = user_importance_df.index + 1

        return {
            'avg_results': avg_results,
            'importance_df': user_importance_df
        }
    except Exception as exc:
        print(f"Error while training user {user}: {exc}")
        raise


def train_user_models(
    dataset_path='selected_users_dataset/full_216features_normalized.pkl',
    results_dir='selected_users_dataset/results',
    results_subdir: Optional[str] = None,
    model_output_dir: Optional[Path] = None,
    save_models: bool = False,
    apply_scaler: bool = True,
    n_trials: int = 50,
    num_boost_round: int = 1000,
    write_outputs: bool = True,
    user_jobs: int = 1,
    threads_per_model: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    results_dir = Path(results_dir).expanduser().resolve()
    dataset_tag = dataset_path.stem
    target_results_dir = results_dir / results_subdir if results_subdir else results_dir

    model_root = _resolve_model_root(dataset_path, model_output_dir).resolve()

    # Load dataset
    with open(dataset_path, 'rb') as f:
        X, y, users, timestamps, feature_names = pickle.load(f)

    if not isinstance(feature_names, list):
        feature_names = list(feature_names)
    feature_names = [str(name) for name in feature_names]
    if len(feature_names) != X.shape[1]:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    users_array = np.asarray(users)

    os.makedirs(target_results_dir, exist_ok=True)

    model_dataset_dir = None
    if save_models:
        model_dataset_dir = model_root / dataset_tag
        if model_dataset_dir.exists():
            shutil.rmtree(model_dataset_dir)
        model_dataset_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    feature_importance_frames: List[pd.DataFrame] = []

    total_cpus = os.cpu_count() or 1
    if threads_per_model is None:
        threads_per_model = max(1, total_cpus // max(1, user_jobs))

    print(f"Using {user_jobs} worker(s); LightGBM threads per model = {threads_per_model}")

    users_unique = np.unique(users_array)

    def submit_user(user_id: str):
        return _train_single_user(
            user=user_id,
            X=X_array,
            y=y_array,
            users=users_array,
            feature_names=feature_names,
            dataset_path=dataset_path,
            dataset_tag=dataset_tag,
            target_results_dir=target_results_dir,
            model_dataset_dir=model_dataset_dir,
            save_models=save_models,
            apply_scaler=apply_scaler,
            n_trials=n_trials,
            num_boost_round=num_boost_round,
            threads_per_model=threads_per_model
        )

    if user_jobs > 1 and len(users_unique) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=user_jobs) as executor:
            future_to_user = {executor.submit(submit_user, user): user for user in users_unique}
            for future in concurrent.futures.as_completed(future_to_user):
                result = future.result()
                if result is None:
                    continue
                results.append(result['avg_results'])
                if result['importance_df'] is not None:
                    feature_importance_frames.append(result['importance_df'])
    else:
        for user in users_unique:
            result = submit_user(user)
            if result is None:
                continue
            results.append(result['avg_results'])
            if result['importance_df'] is not None:
                feature_importance_frames.append(result['importance_df'])

    results_df = pd.DataFrame(results)
    if write_outputs:
        results_csv_path = target_results_dir / 'performance_features.csv'
        results_df.to_csv(results_csv_path, index=False)

    if feature_importance_frames:
        feature_df = pd.concat(feature_importance_frames, ignore_index=True)
        if write_outputs:
            processed_users = feature_df['user'].unique()
            for processed_user in processed_users:
                user_features = feature_df[feature_df['user'] == processed_user].copy()
                user_features = user_features.sort_values('importance', ascending=False)
                column_order = [
                    'feature_index', 'feature_name', 'importance',
                    'importance_std', 'rank', 'normalized_importance'
                ]
                available_columns = [col for col in column_order if col in user_features.columns]
                if available_columns:
                    user_features = user_features[available_columns]
                user_features.to_csv(target_results_dir / f'{processed_user}_feature_importance.csv', index=False)

            feature_summary = feature_df.groupby('feature_name').agg({
                'importance': ['mean', 'std', 'count']
            }).round(4)
            feature_summary.columns = ['mean_importance', 'std_importance', 'user_count']
            feature_summary = feature_summary.reset_index()
            feature_summary = feature_summary.rename(columns={'feature_name': 'feature'})
            feature_summary = feature_summary.sort_values('mean_importance', ascending=False)
            feature_summary.to_csv(target_results_dir / 'feature_summary.csv', index=False)
    else:
        feature_df = pd.DataFrame()
        if write_outputs:
            print("No users produced valid models; skipping feature importance exports.")

    if write_outputs:
        print(f"Training completed. Results saved to {target_results_dir}/")
    return results_df, feature_df


def _infer_raw_dataset_path(dataset_path: Path) -> Optional[Path]:
    if 'normalized' in dataset_path.stem:
        candidate = dataset_path.with_name(dataset_path.name.replace('_normalized', ''))
        if candidate.exists():
            return candidate
    return None


def _filter_dataset_files(
    normalized_path: Path,
    raw_path: Optional[Path],
    selected_users: List[str]
) -> None:
    selected_set = set(selected_users)

    def _filter_and_write(path: Path) -> None:
        with open(path, 'rb') as f:
            X, y, users, timestamps, feature_names = pickle.load(f)

        users_array = np.asarray(users)
        mask = np.isin(users_array, list(selected_set))
        if not np.any(mask):
            raise ValueError(
                f"Filtered dataset '{path}' would be empty; check selection threshold."
            )

        X_filtered = np.asarray(X)[mask]
        y_filtered = np.asarray(y)[mask]
        users_filtered = users_array[mask]
        timestamps_filtered = np.asarray(timestamps)[mask]

        with open(path, 'wb') as f:
            pickle.dump(
                (X_filtered, y_filtered, users_filtered, timestamps_filtered, feature_names),
                f
            )

    _filter_and_write(normalized_path)
    if raw_path and raw_path.exists():
        _filter_and_write(raw_path)


def run_training_pipeline(
    dataset_path: Path,
    raw_dataset_path: Optional[Path],
    results_dir: Path,
    results_subdir: Optional[str],
    model_output_dir: Optional[Path],
    save_models: bool,
    apply_scaler: bool,
    n_trials: int,
    num_boost_round: int,
    selection_threshold: Optional[float],
    user_jobs: int,
    threads_per_model: Optional[int]
):
    dataset_path = dataset_path.expanduser().resolve()
    results_dir = results_dir.expanduser().resolve()
    model_root = _resolve_model_root(dataset_path, model_output_dir)

    raw_dataset_path = raw_dataset_path.expanduser().resolve() if raw_dataset_path else _infer_raw_dataset_path(dataset_path)

    selected_users: List[str] = []

    if selection_threshold is not None:
        print(f"\n=== PRE-SELECTION TRAINING (dataset: {dataset_path}) ===")
        pre_results_df, _ = train_user_models(
            dataset_path=str(dataset_path),
            results_dir=str(results_dir),
            results_subdir=results_subdir,
            model_output_dir=str(model_root),
            save_models=False,
            apply_scaler=apply_scaler,
            n_trials=n_trials,
            num_boost_round=num_boost_round,
            write_outputs=False,
            user_jobs=user_jobs,
            threads_per_model=threads_per_model
        )

        selected_users = (
            pre_results_df.loc[pre_results_df['val_auroc'] >= selection_threshold, 'user']
            .dropna()
            .astype(str)
            .tolist()
        )

        if not selected_users:
            raise ValueError(
                f"No users achieved val_auroc ≥ {selection_threshold}. Unable to continue."
            )

        print(f"\nSelected users (val_auroc ≥ {selection_threshold}): {selected_users}")

        if raw_dataset_path is None:
            raise FileNotFoundError(
                "Raw dataset path could not be inferred. Provide --raw-dataset-path explicitly."
            )

        _filter_dataset_files(dataset_path, raw_dataset_path, selected_users)

        final_results_dir = results_dir / results_subdir if results_subdir else results_dir
        final_results_dir.mkdir(parents=True, exist_ok=True)
        with open(final_results_dir / 'selected_users.json', 'w') as f:
            json.dump({'selection_threshold': selection_threshold, 'users': selected_users}, f, indent=2)

        print(
            f"Filtered datasets written. Normalized: {dataset_path}, Raw: {raw_dataset_path}"
        )

    print(f"\n=== FINAL TRAINING (dataset: {dataset_path}) ===")
    return train_user_models(
        dataset_path=str(dataset_path),
        results_dir=str(results_dir),
        results_subdir=results_subdir,
        model_output_dir=str(model_root),
        save_models=save_models,
        apply_scaler=apply_scaler,
        n_trials=n_trials,
        num_boost_round=num_boost_round,
        write_outputs=True,
        user_jobs=user_jobs,
        threads_per_model=threads_per_model
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-user LightGBM models with optional selection and parallelism.")
    parser.add_argument('--dataset-path', type=Path,
                        default=Path('selected_users_dataset/full_216features_normalized.pkl'),
                        help='Path to the (optionally normalized) dataset pickle used for training.')
    parser.add_argument('--raw-dataset-path', type=Path, default=None,
                        help='Raw dataset pickle to rewrite after selection. If omitted, inferred from dataset path.')
    parser.add_argument('--results-dir', type=Path,
                        default=Path('selected_users_dataset/results'),
                        help='Directory for training outputs.')
    parser.add_argument('--results-subdir', type=str, default=None,
                        help='Optional subdirectory inside results-dir for outputs.')
    parser.add_argument('--model-output-dir', type=Path, default=None,
                        help='Directory root for saving models. Defaults to <dataset_dir>/models.')
    parser.add_argument('--save-models', action='store_true', help='Persist trained LightGBM models to disk.')
    parser.add_argument('--apply-scaler', action='store_true', help='Apply StandardScaler within training.')
    parser.add_argument('--selection-threshold', type=float, default=None,
                        help='If provided, keep only users with val_auroc ≥ threshold, rewrite datasets, and retrain.')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials per user.')
    parser.add_argument('--num-boost-round', type=int, default=1000, help='Number of boosting rounds for final model.')
    parser.add_argument('--user-jobs', type=int, default=-1,
                        help='Number of parallel workers across users.')
    parser.add_argument('--threads-per-model', type=int, default=None,
                        help='Threads used by LightGBM per model (default: auto based on CPU/user-jobs).')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_pipeline(
        dataset_path=args.dataset_path,
        raw_dataset_path=args.raw_dataset_path,
        results_dir=args.results_dir,
        results_subdir=args.results_subdir,
        model_output_dir=args.model_output_dir,
        save_models=args.save_models,
        apply_scaler=args.apply_scaler,
        n_trials=args.n_trials,
        num_boost_round=args.num_boost_round,
        selection_threshold=args.selection_threshold,
        user_jobs=args.user_jobs,
        threads_per_model=args.threads_per_model
    )
