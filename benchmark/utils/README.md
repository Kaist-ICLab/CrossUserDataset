# Shared Benchmark Utilities

This folder documents the **loader and metric contract** that every setting shares. The runnable helpers are distributed between [`../../basemodel-benchmarking/`](../../basemodel-benchmarking/) and [`../../domain_adaptation/`](../../domain_adaptation/); this folder is intentionally light and serves as the conceptual spec.

---

## Data loader contract

All benchmark code must consume the released data via the following contract.

### Input location

`{DATA_ROOT}/{WAVE}/{label}.pkl` where `WAVE ∈ {D1, D2, D3}` and `label` is the lowercase ESM column name (see [`../../data/README.md`](../../data/README.md)).

### Expected return from `pd.read_pickle(path)`

A 5-element tuple `(features, y, groups, t, datetimes)`:

| Element | Type | Description |
|---|---|---|
| `features` | `pd.DataFrame` | Feature matrix, columns follow [`../../data/schema.md`](../../data/schema.md). Includes static `PIF#*` columns. |
| `y` | `np.ndarray` (int8/int64) | Binary label. |
| `groups` | `np.ndarray` (str) | Participant codes for stratified grouping. |
| `t` | `np.ndarray` (float) | Raw (pre-binarisation) label value. |
| `datetimes` | `np.ndarray` | Timestamps for chronological splits. |

See `EDA.utils.load_and_attach` in [`../../EDA/utils.py`](../../EDA/utils.py) for a canonical implementation.

### Required preprocessing steps

All settings should, before training:

1. Drop leakage-prone columns (participant ID, raw label, timestamp).
2. Apply `normalize_label_series` equivalent if the task is cross-wave.
3. For Setting C: apply the alias map and intersect columns across waves (see [`../../preprocessing/feature_alignment.md`](../../preprocessing/feature_alignment.md)).
4. Fit any scaler / imputer on **train only**; apply to val/test unchanged.

---

## Metric contract

| Metric | Role | Library |
|---|---|---|
| AUROC | Primary | `sklearn.metrics.roc_auc_score` |
| Accuracy | Diagnostic | `sklearn.metrics.accuracy_score` |
| Macro-F1 | Diagnostic | `sklearn.metrics.f1_score(average="macro")` |
| Precision | Diagnostic | `sklearn.metrics.precision_score(average="binary")` |
| Recall | Diagnostic | `sklearn.metrics.recall_score(average="binary")` |

All metrics are computed on the setting-specific test set. Model selection uses validation AUROC.

---

## Seed and HPO policy

- Fixed global seed for deterministic comparison (recorded per-run in each setting's output folder).
- Optuna: 30 trials per (model, task, setting).
- Training loop: ≤ 50 epochs, patience-based early stopping.

---

## Output contract

Aggregated leaderboard CSVs live in [`../results/`](../results/). The schema is per-setting:

- Setting A / Setting B (single wave):

  ```
  setting, wave, task, model, family, auroc, auroc_std, acc, macro_f1
  ```

- Setting B category-best (best model per family):

  ```
  setting, wave, task, family, best_model, auroc, auroc_std, acc, macro_f1
  ```

- Setting C (cross-wave, with feature alignment):

  ```
  setting, source, target, task, model, family,
  n_features_after_alignment, auroc, auroc_std, acc, macro_f1
  ```

See [`../results/README.md`](../results/README.md) for the per-column type spec and which files map to which setting.
