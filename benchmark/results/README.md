# Benchmark Results

Committed per-task summary CSVs for every setting. These mirror the leaderboards in Appendix C of the paper so that downstream researchers can compare against our numbers without re-running the full benchmark.

## Files

| File | Setting | Family scope |
|---|---|---|
| `setting_a.csv` | A (per-user temporal, first 30 days) | Baseline + tabular-NN |
| `setting_b_baseline_tabular.csv` | B (within-wave cross-user 5-fold) | Baseline + tabular-NN |
| `setting_b_dg.csv` | B | Domain generalisation |
| `setting_b_da.csv` | B | Domain adaptation |
| `setting_b_category_best.csv` | B | Best model per family per (wave, task) |
| `setting_c_baseline_tabular.csv` | C (cross-wave, 1→1 and 2→1) | Baseline + tabular-NN |
| `setting_c_dg.csv` | C | Domain generalisation |
| `setting_c_da.csv` | C | Domain adaptation |

## Schema

All Setting A / Setting B CSVs share the columns:

```
setting,wave,task,model,family,auroc,auroc_std,acc,macro_f1
```

`setting_b_category_best.csv` reports the best model per family and replaces `model` with `best_model`:

```
setting,wave,task,family,best_model,auroc,auroc_std,acc,macro_f1
```

Setting C adds source / target wave and the post-alignment feature count:

```
setting,source,target,task,model,family,n_features_after_alignment,auroc,auroc_std,acc,macro_f1
```

| Column | Type | Notes |
|---|---|---|
| `setting` | str | One of `A`, `B`, `C`. |
| `wave` | str | `D1`, `D2`, or `D3` (Setting A / B). |
| `source`, `target` | str | Source wave(s) and held-out target wave (Setting C). Single source like `D1`, multi-source joined with `+` (e.g. `D1+D2`). |
| `task` | str | Lowercase ESM label (`valence`, `arousal`, `stress_binary`, `disturbance`, plus D3 affect words). |
| `model` / `best_model` | str | Model name as accepted by `--model` in `basemodel-benchmarking/benchmark.py`. |
| `family` | str | One of `baseline`, `tabular_nn`, `dg`, `da`. |
| `auroc`, `auroc_std` | float | Primary metric: mean and std across folds / seeds. |
| `acc`, `macro_f1` | float | Diagnostic metrics on the test split. |
| `n_features_after_alignment` | int | Common-feature intersection count (Setting C only). |

## Status

These files are **header-only placeholders** until the production sweep is committed. Downstream tooling can be wired up against the schema above before the numbers are available.

## Regenerating

1. Run the relevant sweep (`bash basemodel-benchmarking/scripts/run_setting_a.sh`, `..._b.sh`, `..._c.sh`).
2. Per-experiment metadata is written to `results/records/*.json`.
3. Aggregate the records into the CSVs above (aggregator script to be added).

Per-setting protocol details:

- Setting A: [`../setting_a/README.md`](../setting_a/README.md)
- Setting B: [`../setting_b/README.md`](../setting_b/README.md)
- Setting C: [`../setting_c/README.md`](../setting_c/README.md)
