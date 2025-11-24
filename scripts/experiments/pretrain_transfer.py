#!/usr/bin/env python3
"""CLI for domain adaptation experiments across dataset splits."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain_adaptation.pipeline import ExperimentScenario, run_all_scenarios


def _parse_overrides(entries: Optional[List[str]]) -> Optional[Dict[str, str]]:
    if not entries:
        return None
    mapping: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(f"Invalid override format '{entry}'. Expected NAME=PATH")
        key, value = entry.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain on two datasets, fine-tune on target dataset, and report metrics.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/domain_adaptation"), help="Directory to store cached artifacts.")
    parser.add_argument("--output-csv", type=Path, default=Path("results/domain_adaptation_results.csv"), help="Path to write aggregated CSV results.")
    parser.add_argument("--fine-tune-ratios", type=float, nargs="*", default=[0.0, 0.2, 0.4, 0.6], help="Ratios of target test data used for post-training fine-tuning.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46], help="Random seeds applied to fine-tuning splits and model initialisation.")
    parser.add_argument("--model-types", type=str, nargs="*", default=["tree", "transformer", "cdtrans"], choices=["tree", "transformer", "cdtrans"], help="Model families to evaluate.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation share of the training data for stratified shuffle strategy.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test share of the dataset for stratified shuffle strategy.")
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="stratified_shuffle",
        choices=["stratified_shuffle", "loso", "stratified_group_kfold"],
        help="How to construct target splits.",
    )
    parser.add_argument("--group-folds", type=int, default=5, help="Number of folds for stratified_group_kfold strategy.")
    parser.add_argument("--pretrain-val-ratio", type=float, default=0.1, help="Fraction of combined source data reserved for pretraining validation.")
    parser.add_argument("--target-train-ratio", type=float, help="Direct fraction of target data allocated to training (overrides --val-size).")
    parser.add_argument("--target-val-ratio", type=float, help="Direct fraction of target data allocated to validation.")
    parser.add_argument("--target-test-ratio", type=float, help="Direct fraction of target data allocated to testing.")
    parser.add_argument("--fine-tune-val-ratio", type=float, default=0.2, help="Fraction of fine-tuning samples reserved for validation during adaptation.")
    parser.add_argument("--dataset-override", type=str, nargs="*", help="Override dataset paths (format NAME=PATH).")
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="*",
        help="Explicit scenarios to run (format SOURCE[+SOURCE...]->TARGET, e.g., D-2+D-3->D-4).",
    )
    parser.add_argument(
        "--feature-strategy",
        type=str,
        default="auto",
        choices=["auto", "all_union", "source_union", "intersection"],
        help="How to align feature spaces across datasets. 'auto' drops target-only features when no fine-tuning is used.",
    )
    parser.add_argument(
        "--max-target-shift-std",
        type=float,
        default=15.0,
        help="Drop features whose target mean differs from source mean by more than this many source standard deviations (0 disables).",
    )
    parser.add_argument(
        "--feature-clip-value",
        type=float,
        default=5.0,
        help="Clip standardized features into [-value, value] before feeding models (set <=0 to disable).",
    )
    parser.add_argument(
        "--min-source-feature-frac",
        type=float,
        default=0.005,
        help="Drop features that are non-zero in fewer than this fraction of combined source samples (default 0.5%).",
    )
    parser.add_argument(
        "--min-source-feature-std",
        type=float,
        default=0.0,
        help="Optional minimum standard deviation (computed over source samples) required to keep a feature.",
    )
    parser.add_argument(
        "--feature-zero-tol",
        type=float,
        default=1e-6,
        help="Values with absolute magnitude <= this tolerance count as zero for sparsity filtering.",
    )
    parser.add_argument(
        "--max-feature-correlation",
        type=float,
        default=0.0,
        help="Drop features whose absolute correlation within the same prefix exceeds this threshold (0 disables).",
    )
    parser.add_argument(
        "--correlation-sample-rows",
        type=int,
        default=2000,
        help="Number of rows to sample from each source dataset when computing correlation pruning.",
    )
    parser.add_argument(
        "--max-class-conditional-shift",
        type=float,
        default=0.0,
        help="Drop features whose positive-class activation differs between source and target by more than this fraction.",
    )
    parser.add_argument(
        "--class-shift-max-samples",
        type=int,
        default=5000,
        help="Maximum rows from source/target used to estimate class-conditional shifts.",
    )
    parser.add_argument(
        "--l1-feature-keep-ratio",
        type=float,
        default=0.0,
        help="If >0, keep only this ratio of features with highest L1-logistic importance (preprocessing step).",
    )
    parser.add_argument(
        "--l1-max-samples",
        type=int,
        default=5000,
        help="Maximum rows used when fitting the L1 logistic selector.",
    )
    parser.add_argument(
        "--include-target-in-scaler",
        dest="include_target_in_scaler",
        action="store_true",
        help="Fit the StandardScaler on both sources and (unlabeled) target features.",
    )
    parser.add_argument(
        "--exclude-target-in-scaler",
        dest="include_target_in_scaler",
        action="store_false",
        help="Fit the StandardScaler using sources only (previous behaviour).",
    )
    parser.set_defaults(include_target_in_scaler=True)
    parser.add_argument(
        "--transformer-option",
        action="append",
        metavar="KEY=VALUE",
        help="Override TransformerConfig fields (repeat flag for multiple entries; VALUE parsed with literal_eval when possible).",
    )

    args = parser.parse_args()

    overrides = _parse_overrides(args.dataset_override)
    scenario_specs: Optional[List[ExperimentScenario]] = None
    if args.scenarios:
        scenario_specs = []
        for entry in args.scenarios:
            if "->" not in entry:
                raise argparse.ArgumentTypeError(f"Invalid scenario '{entry}'. Expected format SOURCE[+SOURCE...]->TARGET")
            left, right = entry.split("->", 1)
            right = right.strip()
            if not right:
                raise argparse.ArgumentTypeError(f"Scenario target missing in '{entry}'")
            sources = [src.strip() for src in left.split("+") if src.strip()]
            if not sources:
                raise argparse.ArgumentTypeError(f"Scenario sources missing in '{entry}'")
            scenario_specs.append(ExperimentScenario(target=right, sources=sources))

    transformer_overrides: Optional[Dict[str, object]] = None
    if args.transformer_option:
        transformer_overrides = {}
        for entry in args.transformer_option:
            if "=" not in entry:
                raise argparse.ArgumentTypeError(f"Invalid transformer override '{entry}'. Expected KEY=VALUE")
            key, value = entry.split("=", 1)
            key = key.strip()
            if not key:
                raise argparse.ArgumentTypeError(f"Transformer override key missing in '{entry}'")
            raw_value = value.strip()
            try:
                parsed = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                parsed = raw_value
            transformer_overrides[key] = parsed
    ratios = sorted(set(max(0.0, min(float(r), 0.99)) for r in args.fine_tune_ratios))
    seeds = sorted(set(int(s) for s in args.seeds))

    df = run_all_scenarios(
        cache_dir=args.cache_dir,
        overrides=overrides,
        fine_tune_ratios=ratios,
        seeds=seeds,
        model_types=args.model_types,
        val_size=max(0.0, min(args.val_size, 0.8)),
        test_size=max(0.0, min(args.test_size, 0.8)),
        pretrain_val_ratio=max(0.0, min(args.pretrain_val_ratio, 0.5)),
        fine_tune_val_ratio=max(0.0, min(args.fine_tune_val_ratio, 0.5)),
        target_train_ratio=args.target_train_ratio,
        target_val_ratio=args.target_val_ratio,
        target_test_ratio=args.target_test_ratio,
        split_strategy=args.split_strategy,
        group_folds=max(2, args.group_folds),
        output_csv=args.output_csv,
        feature_strategy=args.feature_strategy,
        min_source_feature_frac=max(0.0, args.min_source_feature_frac),
        min_source_feature_std=max(0.0, args.min_source_feature_std),
        feature_zero_tolerance=max(1e-12, args.feature_zero_tol),
        include_target_in_scaler=args.include_target_in_scaler,
        feature_clip_value=None if args.feature_clip_value <= 0 else float(args.feature_clip_value),
        transformer_overrides=transformer_overrides,
        max_target_shift_std=max(0.0, args.max_target_shift_std),
        max_feature_correlation=max(0.0, args.max_feature_correlation),
        correlation_sample_rows=max(0, args.correlation_sample_rows),
        max_class_conditional_shift=max(0.0, args.max_class_conditional_shift),
        class_shift_max_samples=max(0, args.class_shift_max_samples),
        l1_feature_keep_ratio=max(0.0, min(1.0, args.l1_feature_keep_ratio)),
        l1_max_samples=max(0, args.l1_max_samples),
        scenario_specs=scenario_specs,
    )

    print(f"Saved {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
