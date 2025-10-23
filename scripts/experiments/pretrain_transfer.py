#!/usr/bin/env python3
"""CLI for domain adaptation experiments across dataset splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain_adaptation.pipeline import run_all_scenarios


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
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45], help="Random seeds applied to fine-tuning splits and model initialisation.")
    parser.add_argument("--model-types", type=str, nargs="*", default=["tree", "transformer"], choices=["tree", "transformer"], help="Model families to evaluate.")
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

    args = parser.parse_args()

    overrides = _parse_overrides(args.dataset_override)
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
    )

    print(f"Saved {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
