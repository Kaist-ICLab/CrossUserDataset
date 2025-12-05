import os
import logging
import random
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
import torch


# ==========================
# Global config
# ==========================

RANDOM_STATE = 42
SEEDS = [0, 1, 2, 3, 4]
N_SPLITS = 5
TARGET_DATASET = "D-4"  # change to "D-2" or "D-4" if desired


DEBUG_N_SAMPLES = None  # set to None to use full dataset

META_COLS = ["META#dataset", "PIF#participantID", "PIF#time_offset", "PIF#timestamp"]
DROP_COLS = [
    "PIF#participantID",
    "PIF#timestamp",
    "PIF#participationStartTimestamp",
    "PIF#time_offset",
    "META#dataset",
    "__src",
    "PIF#stress_label",
]

METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "roc_auc",
    "brier_score",
]

# Reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Logger placeholder
LOGGER: Optional[logging.Logger] = None


# ==========================
# Utility functions
# ==========================


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "basemodel_benchmark.log")

    logger = logging.getLogger("basemodel_benchmark")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if main() is called multiple times
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Logging initialized. Log file: %s", log_path)
    return logger


def load_and_attach(path: str, dataset_tag: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load pickle and attach meta columns, mirroring the notebook logic."""
    df, y, groups, t, datetimes = pd.read_pickle(path)

    meta = pd.DataFrame(
        {
            "META#dataset": dataset_tag,
            "PIF#participantID": groups,
            "PIF#time_offset": t,
            "PIF#timestamp": datetimes,
        }
    )

    assert len(df) == len(meta), f"Row mismatch in {dataset_tag}"

    out = pd.concat([meta.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    feature_cols = sorted(c for c in out.columns if c not in META_COLS)
    X = out[META_COLS + feature_cols]
    y = pd.Series(y, name="PIF#stress_label")
    return X, y


def load_datasets(base_dir: str) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    files = {
        "D-2": os.path.join(base_dir, "D-2", "Intermediate", "stress_binary_personal-full.pkl"),
        "D-3": os.path.join(base_dir, "D-3", "Intermediate", "stress_binary_personal-full_D#3.pkl"),
        "D-4": os.path.join(base_dir, "D-4", "Intermediate", "stress_binary_personal-full.pkl"),
    }

    datasets = {}
    for tag, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing pickle for {tag}: {path}")
        X, y = load_and_attach(path, tag)
        datasets[tag] = (X, y)
    return datasets


def get_common_cols(dfs: List[pd.DataFrame]) -> List[str]:
    feature_sets = [set(df.columns) - set(META_COLS) for df in dfs]
    common = sorted(set.intersection(*feature_sets))
    return META_COLS + common


def prepare_target_dataset(datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]]):
    # Align columns across D-2/3/4 as in notebook
    df_1, y_1 = datasets["D-2"]
    df_2, y_2 = datasets["D-3"]
    df_3, y_3 = datasets["D-4"]

    common_cols = get_common_cols([df_1, df_2, df_3])

    df_1_over = df_1[common_cols]
    df_2_over = df_2[common_cols]
    df_3_over = df_3[common_cols]

    if TARGET_DATASET == "D-2":
        df_over, y = df_1_over, y_1
    elif TARGET_DATASET == "D-3":
        df_over, y = df_2_over, y_2
    elif TARGET_DATASET == "D-4":
        df_over, y = df_3_over, y_3
    else:
        raise ValueError(f"Unknown TARGET_DATASET: {TARGET_DATASET}")

    # Features X, labels y, plus groups and timestamps for splits
    groups = df_over["PIF#participantID"].astype(str).values
    timestamps = pd.to_datetime(df_over["PIF#timestamp"])  # ensure datetime

    X = df_over.drop(columns=DROP_COLS, errors="ignore")

    if LOGGER:
        LOGGER.info(
            "Prepared dataset %s: X shape=%s, y len=%d, n_participants=%d",
            TARGET_DATASET,
            X.shape,
            len(y),
            len(np.unique(groups)),
        )

    return X, y, groups, timestamps


def get_models() -> Dict[str, object]:
    models = {}

    # 1) TabPFN
    models["TabPFN"] = TabPFNClassifier(
        device="cuda",
        ignore_pretraining_limits=True,
    )

    # 2) MLP (with scaling in a pipeline)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        alpha=1e-4,
        random_state=RANDOM_STATE,
    )

    models["MLP"] = make_pipeline(
        StandardScaler(),
        mlp,
    )

    # 3) XGBoost
    models["XGBoost"] = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    return models


def _extract_positive_scores(y_proba: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if y_proba is None:
        return None
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        return y_proba
    if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
        return y_proba[:, 1]
    return None


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """Compute core metrics + confusion matrix for binary classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, float] = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    y_score = _extract_positive_scores(y_proba)

    # ROC AUC & Brier score (binary only, with valid scores)
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            metrics["roc_auc"] = np.nan
        try:
            metrics["brier_score"] = brier_score_loss(y_true, y_score)
        except Exception:
            metrics["brier_score"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["brier_score"] = np.nan

    cm = None
    if len(np.unique(y_true)) == 2:
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        except Exception:
            cm = None

    return metrics, cm


def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups_test: Optional[np.ndarray] = None,
    return_details: bool = False,
):
    """Fit model, predict, compute metrics, and optionally return per-sample details.

    Returns (metrics_dict, extras_dict).
    """
    global LOGGER

    if LOGGER:
        LOGGER.info("Training %s (n_train=%d, n_test=%d)", name, len(X_train), len(X_test))

    try:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)

        metrics, cm = compute_binary_classification_metrics(y_test, y_pred, y_proba)

        extras = {
            "confusion_matrix": cm,
        }

        if return_details:
            extras.update(
                {
                    "y_true": np.asarray(y_test),
                    "y_pred": np.asarray(y_pred),
                    "y_score": _extract_positive_scores(y_proba),
                    "groups_test": np.asarray(groups_test) if groups_test is not None else None,
                }
            )

        if LOGGER:
            LOGGER.info("Finished %s | metrics=%s", name, {k: round(v, 4) if v == v else None for k, v in metrics.items()})

        return metrics, extras

    except Exception as e:
        if LOGGER:
            LOGGER.exception("Error training %s: %s", name, e)
        # Fill metrics with NaNs on failure
        metrics = {m: np.nan for m in METRIC_NAMES}
        extras = {
            "confusion_matrix": None,
        }
        if return_details:
            extras.update({"y_true": None, "y_pred": None, "y_score": None, "groups_test": None})
        return metrics, extras


# ==========================
# Split strategies
# ==========================


def run_random_kfold(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    seeds: List[int],
) -> pd.DataFrame:
    """Random Stratified K-Fold, repeated with multiple seeds.

    Returns a DataFrame with per-fold metrics.
    """
    rows = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        fold_idx = 0
        for train_idx, test_idx in skf.split(X, y):
            fold_idx += 1
            if LOGGER:
                LOGGER.info("Random K-Fold | seed=%d, fold=%d/%d", seed, fold_idx, n_splits)

            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            for model_name, base_model in models.items():
                model = clone(base_model)
                metrics, extras = evaluate_model(
                    f"{model_name} (random, seed={seed}, fold={fold_idx})",
                    model,
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    return_details=False,
                )

                cm = extras.get("confusion_matrix")
                tn = fp = fn = tp = np.nan
                if cm is not None and cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()

                row = {
                    "split_type": "random_kfold",
                    "model": model_name,
                    "seed": seed,
                    "fold": fold_idx,
                    "n_train": len(X_tr),
                    "n_test": len(X_te),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
                row.update(metrics)
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


def run_group_kfold(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    n_splits: int,
) -> pd.DataFrame:
    """Group K-Fold using participant ID as group label."""
    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    fold_idx = 0
    for train_idx, test_idx in gkf.split(X, y, groups):
        fold_idx += 1
        if LOGGER:
            LOGGER.info("Group K-Fold | fold=%d/%d", fold_idx, n_splits)

        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        for model_name, base_model in models.items():
            model = clone(base_model)
            metrics, extras = evaluate_model(
                f"{model_name} (group, fold={fold_idx})",
                model,
                X_tr,
                y_tr,
                X_te,
                y_te,
                return_details=False,
            )

            cm = extras.get("confusion_matrix")
            tn = fp = fn = tp = np.nan
            if cm is not None and cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

            row = {
                "split_type": "group_kfold",
                "model": model_name,
                "fold": fold_idx,
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_time_based_masks(groups: np.ndarray, timestamps: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Build boolean train/test masks: per user, first 70% by time = train, last 30% = test."""
    n = len(groups)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    groups_arr = np.asarray(groups)
    ts = pd.to_datetime(timestamps).values

    unique_users = np.unique(groups_arr)

    for user in unique_users:
        idx = np.where(groups_arr == user)[0]
        if len(idx) < 2:
            continue  # can't split

        # sort by timestamp
        idx_sorted = idx[np.argsort(ts[idx])]
        n_user = len(idx_sorted)

        # ensure at least 1 in train and 1 in test
        raw_cut = int(np.floor(0.7 * n_user))
        cut = min(max(raw_cut, 1), n_user - 1)

        train_idx = idx_sorted[:cut]
        test_idx = idx_sorted[cut:]

        train_mask[train_idx] = True
        test_mask[test_idx] = True

    return train_mask, test_mask


def run_time_split_per_user(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    timestamps: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split per user, then per-user AUROC aggregation.

    Returns:
        per_user_df: rows = (model, participant, per-user AUROC, n_test)
        global_df: one row per model with global metrics on all test samples.
    """
    train_mask, test_mask = build_time_based_masks(groups, timestamps)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    groups_test = np.asarray(groups)[test_mask]

    if LOGGER:
        LOGGER.info(
            "Time-based split: n_train=%d, n_test=%d, n_users_test=%d",
            len(X_train),
            len(X_test),
            len(np.unique(groups_test)),
        )

    per_user_rows = []
    global_rows = []

    for model_name, base_model in models.items():
        model = clone(base_model)
        metrics, extras = evaluate_model(
            f"{model_name} (time-split)",
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            groups_test=groups_test,
            return_details=True,
        )

        cm = extras.get("confusion_matrix")
        tn = fp = fn = tp = np.nan
        if cm is not None and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

        # Global metrics row
        global_row = {
            "split_type": "time_split_global",
            "model": model_name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        global_row.update(metrics)
        global_rows.append(global_row)

        # Per-user AUROC
        y_true_all = extras.get("y_true")
        y_score_all = extras.get("y_score")
        groups_all = extras.get("groups_test")

        if y_true_all is None or y_score_all is None or groups_all is None:
            continue

        y_true_all = np.asarray(y_true_all)
        y_score_all = np.asarray(y_score_all)
        groups_all = np.asarray(groups_all)

        for user in np.unique(groups_all):
            mask = groups_all == user
            if mask.sum() < 2:
                continue

            y_true_u = y_true_all[mask]
            y_score_u = y_score_all[mask]

            # require both classes present for AUROC
            if len(np.unique(y_true_u)) < 2:
                auroc_u = np.nan
            else:
                try:
                    auroc_u = roc_auc_score(y_true_u, y_score_u)
                except Exception:
                    auroc_u = np.nan

            per_user_rows.append(
                {
                    "split_type": "time_split_per_user",
                    "model": model_name,
                    "participant": user,
                    "roc_auc": auroc_u,
                    "n_test": int(mask.sum()),
                }
            )

    per_user_df = pd.DataFrame(per_user_rows)
    global_df = pd.DataFrame(global_rows)
    return per_user_df, global_df


# ==========================
# Plotting helpers
# ==========================


def plot_auroc_distributions(df: pd.DataFrame, split_label: str, out_path: str):
    """Save boxplot of AUROC distributions per model for a given split."""
    if "roc_auc" not in df.columns or df.empty:
        return

    models = sorted(df["model"].unique())
    data = [df.loc[df["model"] == m, "roc_auc"].dropna().values for m in models]

    if all(len(d) == 0 for d in data):
        return

    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=models, showmeans=True)
    plt.ylabel("AUROC")
    plt.title(f"AUROC distribution - {split_label}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ==========================
# Summary helpers
# ==========================


def summarize_auroc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "roc_auc" not in df.columns:
        return pd.DataFrame()
    summary = (
        df.groupby("model")["roc_auc"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "auroc_mean", "std": "auroc_std"})
    )
    return summary


def build_final_summary_table(
    random_df: pd.DataFrame,
    group_df: pd.DataFrame,
    time_user_df: pd.DataFrame,
) -> pd.DataFrame:
    rand_sum = summarize_auroc(random_df)
    grp_sum = summarize_auroc(group_df)
    time_sum = summarize_auroc(time_user_df)

    all_models = sorted(
        set(rand_sum.index).union(grp_sum.index).union(time_sum.index)
    )

    rows = []
    for m in all_models:
        r_m = rand_sum.loc[m] if m in rand_sum.index else pd.Series()
        g_m = grp_sum.loc[m] if m in grp_sum.index else pd.Series()
        t_m = time_sum.loc[m] if m in time_sum.index else pd.Series()

        def fmt(row):
            if "auroc_mean" not in row or "auroc_std" not in row:
                return ""
            mean = row["auroc_mean"]
            std = row["auroc_std"]
            if pd.isna(mean):
                return ""
            return f"{mean:.3f} +/- {std:.3f}" if not pd.isna(std) else f"{mean:.3f}"

        rows.append(
            {
                "Model": m,
                "Random KFold AUROC (mean +/- std)": fmt(r_m),
                "Group KFold AUROC (mean +/- std)": fmt(g_m),
                "Time-Split AUROC (mean +/- std)": fmt(t_m),
            }
        )

    final_df = pd.DataFrame(rows).set_index("Model")
    return final_df


def write_text_summary(final_df: pd.DataFrame, out_path: str):
    """Write a short human-readable summary about difficulty of splits and TabPFN."""
    lines = []

    if final_df.empty:
        lines.append("No results available to summarize.")
    else:
        # Determine which split is hardest: lowest average AUROC across models
        cols = [
            "Random KFold AUROC (mean +/- std)",
            "Group KFold AUROC (mean +/- std)",
            "Time-Split AUROC (mean +/- std)",
        ]

        split_means = {}
        for col in cols:
            vals = []
            for s in final_df[col].dropna():
                try:
                    mean_str = s.split(" +/- ")[0]
                    vals.append(float(mean_str))
                except Exception:
                    continue
            if vals:
                split_means[col] = float(np.mean(vals))

        if split_means:
            hardest = min(split_means, key=split_means.get)
            easiest = max(split_means, key=split_means.get)

            lines.append("Which split is hardest?")
            lines.append(f"- Hardest split: {hardest} (lowest mean AUROC across models).")
            lines.append(f"- Easiest split: {easiest} (highest mean AUROC across models).")
            lines.append("")

        # Does TabPFN still lead?
        if "TabPFN" in final_df.index:
            lines.append("TabPFN vs. other models:")
            tabpfn_row = final_df.loc["TabPFN"]
            for col in cols:
                val = tabpfn_row.get(col, "")
                lines.append(f"- {col}: {val}")
            lines.append("")
        else:
            lines.append("TabPFN results are not available (likely failed or was skipped).\n")

        # How far performance drops from random -> group/time
        lines.append("Performance drops relative to Random KFold (per model):")
        for model, row in final_df.iterrows():
            def parse_mean(s: str) -> Optional[float]:
                if not isinstance(s, str) or not s:
                    return None
                try:
                    return float(s.split(" +/- ")[0])
                except Exception:
                    return None

            r = parse_mean(row.get("Random KFold AUROC (mean +/- std)", ""))
            g = parse_mean(row.get("Group KFold AUROC (mean +/- std)", ""))
            t = parse_mean(row.get("Time-Split AUROC (mean +/- std)", ""))

            parts = [f"Model {model}:"]
            if r is not None and g is not None:
                parts.append(f"Group KFold ΔAUROC: {r - g:.3f}")
            if r is not None and t is not None:
                parts.append(f"Time-Split ΔAUROC: {r - t:.3f}")

            lines.append("- " + "; ".join(parts))

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    if LOGGER:
        LOGGER.info("Wrote text summary to %s", out_path)


# ==========================
# Main entry point
# ==========================


def main():
    # Determine script directory and project root.
    # We want:
    # - data (D-2/D-3/D-4 pickles) from the project root
    # - outputs (basemodel_benchmark_outputs) inside the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == ".ipynb_checkpoints":
        project_root = os.path.dirname(script_dir)
    else:
        project_root = script_dir

    # Outputs go next to this script (inside .ipynb_checkpoints if that is where it lives)
    output_dir = os.path.join(script_dir, "basemodel_benchmark_outputs")
    csv_dir = os.path.join(output_dir, "csv")
    plots_dir = os.path.join(output_dir, "plots")
    logs_dir = os.path.join(output_dir, "logs")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    global LOGGER
    LOGGER = setup_logging(logs_dir)

    LOGGER.info("Starting baseline model benchmarking for dataset %s", TARGET_DATASET)

    # Load data from project root (where D-2/D-3/D-4 live)
    datasets = load_datasets(project_root)
    X, y, groups, timestamps = prepare_target_dataset(datasets)

    # ----------------------------------------------
    # Debug mode: reduce dataset size for quick pipeline debug
    # ----------------------------------------------
    if DEBUG_N_SAMPLES is not None and len(X) > DEBUG_N_SAMPLES:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X), size=DEBUG_N_SAMPLES, replace=False)

        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        groups = np.asarray(groups)[idx]
        timestamps = timestamps.iloc[idx].reset_index(drop=True)

        LOGGER.info(
            "Using debug subset: %d samples (out of %d)",
            DEBUG_N_SAMPLES,
            len(datasets[TARGET_DATASET][0]),
        )
    # ----------------------------------------------


    # Define models
    models = get_models()

    # A. Random K-Fold (with multiple seeds)
    random_df = run_random_kfold(models, X, y, n_splits=N_SPLITS, seeds=SEEDS)
    random_csv = os.path.join(csv_dir, "random_kfold_results.csv")
    random_df.to_csv(random_csv, index=False)
    LOGGER.info("Saved random K-fold results to %s", random_csv)

    plot_auroc_distributions(
        random_df,
        split_label="Random K-Fold (all seeds)",
        out_path=os.path.join(plots_dir, "auroc_random_kfold.png"),
    )

    # B. Group K-Fold (participant-level)
    group_df = run_group_kfold(models, X, y, groups, n_splits=N_SPLITS)
    group_csv = os.path.join(csv_dir, "group_kfold_results.csv")
    group_df.to_csv(group_csv, index=False)
    LOGGER.info("Saved group K-fold results to %s", group_csv)

    plot_auroc_distributions(
        group_df,
        split_label="Group K-Fold (participant)",
        out_path=os.path.join(plots_dir, "auroc_group_kfold.png"),
    )

    # C. Time-based split per user
    time_user_df, time_global_df = run_time_split_per_user(models, X, y, groups, timestamps)
    time_user_csv = os.path.join(csv_dir, "time_split_per_user_results.csv")
    time_global_csv = os.path.join(csv_dir, "time_split_global_results.csv")
    time_user_df.to_csv(time_user_csv, index=False)
    time_global_df.to_csv(time_global_csv, index=False)
    LOGGER.info("Saved time-split per-user results to %s", time_user_csv)
    LOGGER.info("Saved time-split global results to %s", time_global_csv)

    plot_auroc_distributions(
        time_user_df,
        split_label="Time-based split (per user)",
        out_path=os.path.join(plots_dir, "auroc_time_split_per_user.png"),
    )

    # Build final summary table
    final_df = build_final_summary_table(random_df, group_df, time_user_df)
    final_csv = os.path.join(csv_dir, "final_summary_table.csv")
    final_df.to_csv(final_csv)
    LOGGER.info("Saved final summary table to %s", final_csv)

    # Write short text summary
    text_summary_path = os.path.join(output_dir, "summary.txt")
    write_text_summary(final_df, text_summary_path)

    LOGGER.info("Benchmarking completed.")


if __name__ == "__main__":
    main()
