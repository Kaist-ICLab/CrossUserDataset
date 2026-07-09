import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_logger import BenchmarkLogger, evaluate_extended
from data_loader import (
    BenchmarkDataset,
    filter_split_for_label_diversity,
    rebinarize_personal_train_only,
)
from hparams_registry import get_hparams

from domain_adaptation.models.da_models import (
    ADDA,
    CBST,
    CDAN,
    CGDM,
    DAN,
    DANN,
    JAN,
    MCC,
    MCD,
    SHOT,
    DeepCORAL,
    MCDInferenceWrapper,
    train_adda,
    train_cbst,
    train_cdan,
    train_cgdm,
    train_dan,
    train_dann,
    train_deepcoral,
    train_jan,
    train_mcc,
    train_mcd,
    train_shot,
)
from domain_adaptation.models.domainbed_algos import (
    CSD,
)
from domain_adaptation.models.domainbed_algos import ERM as DG_ERM
from domain_adaptation.models.domainbed_algos import (
    IRM,
    MASF,
    MLDG,
    Fish,
    GroupDRO,
    MixStyle,
    SagNet,
    VREx,
    train_dg_model,
)
from models import (
    MLP,
    DeepCTRWrapper,
    LightGBMWrapper,
    ResNet,
    TabNetWrapper,
    WidedeepWrapper,
    XGBoostWrapper,
    attach_training_metadata,
    evaluate_model,
    train_torch_model,
)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

BASE_DATA_DIR = str((Path(__file__).resolve().parent.parent / "data").resolve())
FIXED_BATCH_SIZE = 16
DEFAULT_BATCH_SIZE = 512
MODEL_DEFAULT_BATCH_SIZES = {
    "TabNet": 512,
    "DCN": 512,
    "AutoInt": 512,
    "CGDM": 512,
    "SAINT": 512,
    "TabTransformer": 512,
    "FTTransformer": 512,
}


def release_torch_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_default_batch_size(model_name: str) -> int:
    return MODEL_DEFAULT_BATCH_SIZES.get(model_name, DEFAULT_BATCH_SIZE)


PROGRESS_COLUMNS = [
    "Dataset",
    "Label",
    "Model",
    "Backbone",
    "Seed",
    "Fold",
    "Phase",
    "Trial",
    "Train_Accuracy",
    "Train_AUROC",
    "Train_F1",
    "Train_Precision",
    "Train_Recall",
    "Val_Accuracy",
    "Val_AUROC",
    "Val_F1",
    "Val_Precision",
    "Val_Recall",
    "Test_Accuracy",
    "Test_F1",
    "Test_AUROC",
    "Test_Precision",
    "Test_Recall",
    "Train_Samples",
    "Val_Samples",
    "Test_Samples",
    "Train_Users",
    "Val_Users",
    "Test_Users",
    "Train_PosRatio",
    "Val_PosRatio",
    "Test_PosRatio",
    "Best_Epoch",
    "Early_Stopped",
    "Seed_Count",
    "HPO_Best_AUROC",
    "HPO_Planned_Trials",
    "HPO_Completed_Trials",
    "Configured_Max_Epochs",
    "Configured_Default_Batch_Size",
    "Selected_Batch_Size",
    "Selected_LR",
    "Total_Wall_S",
    "HPO_Wall_S",
    "Train_Wall_S",
    "Eval_Wall_S",
    "Train_GPU_Hours",
    "Eval_GPU_Hours",
    "Total_GPU_Hours",
    "Device_Name",
    "GPU_Model",
    "GPU_Count",
    "GPU_Total_VRAM_GB",
    "CPU_Model",
    "RAM_GB",
    "Peak_GPU_MB",
    "Peak_CPU_MB",
    "Param_Count",
    "Trainable_Param_Count",
    "Artifact_Size_MB",
    "Inference_Batch_Size",
    "Inference_Latency_MS",
    "Inference_Throughput_SPS",
    "FLOPs",
    "MACs",
    "Energy_KWh",
    "Carbon_KgCO2eq",
    "Hparams_JSON",
    "Experiment_ID",
]


def append_row(path, row, columns):
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(path, mode="a", header=header, index=False)


def parse_seeds(seed_values):
    return [int(seed) for seed in seed_values]


def make_groupwise_val_split(train_idx, labels, groups, seed=42, max_splits=5):
    train_idx = np.asarray(train_idx, dtype=int)
    train_labels = labels[train_idx]
    train_groups = groups[train_idx]
    unique_groups = np.unique(train_groups)
    if unique_groups.size < 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        dummy = np.zeros_like(train_labels)
        train_rel, val_rel = next(sss.split(dummy, train_labels))
        return train_idx[train_rel], train_idx[val_rel]
    n_splits = int(min(max_splits, unique_groups.size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros_like(train_labels)
    try:
        train_rel, val_rel = next(splitter.split(dummy, train_labels, train_groups))
    except ValueError:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_rel, val_rel = next(sss.split(dummy, train_labels))
    return train_idx[train_rel], train_idx[val_rel]


def make_temporal_global_split(
    labels: np.ndarray,
    users: np.ndarray,
    timestamps: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    drop_days: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    labels = np.asarray(labels)
    users = np.asarray(users)
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Temporal split failed: {bad} timestamps could not be parsed.")

    train_indices = []
    val_indices = []
    test_indices = []

    unique_users = np.unique(users)
    all_classes = np.unique(labels)
    min_required_classes = min(2, len(all_classes))
    reason_counts: Dict[str, int] = {
        "kept_users": 0,
        "dropped_insufficient_samples": 0,
        "dropped_insufficient_label_diversity": 0,
    }

    for user in unique_users:
        user_idx = np.where(users == user)[0]
        if user_idx.size == 0:
            continue

        user_ts = ts.iloc[user_idx]
        order = np.argsort(user_ts.astype("int64").to_numpy())
        user_idx_sorted = user_idx[order]
        user_ts_sorted = user_ts.iloc[order]

        start_time = user_ts_sorted.iloc[0]
        cutoff_time = start_time + pd.Timedelta(days=drop_days)
        within_cutoff = (user_ts_sorted <= cutoff_time).to_numpy()
        user_idx_window = user_idx_sorted[within_cutoff]

        n = user_idx_window.size
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        if n_train < 1 or n_val < 1 or n_test < 1:
            reason_counts["dropped_insufficient_samples"] += 1
            continue

        tr = user_idx_window[:n_train]
        va = user_idx_window[n_train : n_train + n_val]
        te = user_idx_window[n_train + n_val :]

        if (
            np.unique(labels[tr]).size < min_required_classes
            or np.unique(labels[va]).size < min_required_classes
            or np.unique(labels[te]).size < min_required_classes
        ):
            reason_counts["dropped_insufficient_label_diversity"] += 1
            continue

        train_indices.extend(tr.tolist())
        val_indices.extend(va.tolist())
        test_indices.extend(te.tolist())
        reason_counts["kept_users"] += 1

    if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            "Temporal split produced empty train/val/test after user filtering. "
            "Try adjusting drop window or label constraints."
        )

    return (
        np.asarray(train_indices, dtype=int),
        np.asarray(val_indices, dtype=int),
        np.asarray(test_indices, dtype=int),
        reason_counts,
    )


def get_args():
    parser = argparse.ArgumentParser(description="Run Within-Dataset Benchmark")
    parser.add_argument("--dataset", type=str, required=True, choices=["D-1", "D-2", "D-3"])
    parser.add_argument("--label", type=str, default="stress_binary")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "XGB",
            "LGB",
            "MLP",
            "ResNet",
            "DANN",
            "CDAN",
            "DAN",
            "DeepCORAL",
            "MCC",
            "ADDA",
            "MCD",
            "JAN",
            "SHOT",
            "CBST",
            "CGDM",
            "TabNet",
            "SAINT",
            "TabTransformer",
            "FTTransformer",
            "TFTransformer",
            "TF-transformer",
            "DCN",
            "AutoInt",
            "IRM",
            "VREx",
            "GroupDRO",
            "MixStyle",
            "ERM_DG",
            "MLDG",
            "MASF",
            "Fish",
            "CSD",
            "SagNet",
        ],
    )
    parser.add_argument(
        "--backbone", type=str, default="MLP", choices=["MLP", "ResNet", "Transformer"]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=FIXED_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output", type=str, default="results/benchmark_results_da_hpo.csv")
    parser.add_argument("--hpo_trials", type=int, default=30)
    parser.add_argument("--hpo_mode", type=str, default="fold1", choices=["fold1", "cv", "nested"])
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--efficient_attention", action="store_true")
    parser.add_argument("--uda", action="store_true")
    parser.add_argument(
        "--max_folds", type=int, default=None, help="Limit number of folds (e.g. 1 for quick debug)"
    )
    parser.add_argument(
        "--epochs_override", type=int, default=None, help="Override epochs for quick debug runs"
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="group_kfold",
        choices=["group_kfold", "temporal"],
        help="Data split strategy: existing StratifiedGroupKFold or per-user temporal split baseline",
    )
    parser.add_argument("--temporal_train_ratio", type=float, default=0.6)
    parser.add_argument("--temporal_val_ratio", type=float, default=0.2)
    parser.add_argument("--temporal_drop_days", type=int, default=30)
    parser.add_argument(
        "--rebin_personal",
        type=str,
        default="train_median",
        choices=["none", "train_median", "train_mean"],
        help=(
            "Tier A only: re-derive per-user binary stress label using only the training "
            "split. Avoids the leakage in stress_binary_personal-full_*.pkl, where the "
            'per-user threshold was computed over the full user history. "none" keeps the '
            "precomputed (leaky) labels for backward compatibility."
        ),
    )
    args = parser.parse_args()
    if args.batch_size == FIXED_BATCH_SIZE:
        args.batch_size = get_default_batch_size(args.model)
    return args


def train_model(
    args,
    X_train,
    y_train,
    d_train,
    X_val,
    y_val,
    d_val,
    input_dim,
    num_classes,
    num_domains,
    hparams,
    seed=42,
    patience=20,
    X_target=None,
):

    logger.info(
        f"  [DEBUG] train_model params: Backbone={args.backbone}, Model={args.model}, LR={hparams.get('lr')}"
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = None

    backbone = hparams.get("backbone", args.backbone)
    lr = hparams.get("lr", args.lr)
    batch_size = args.batch_size
    epochs = args.epochs_override if args.epochs_override else args.epochs
    dropout = hparams.get("dropout", 0.3)
    hidden_dim = hparams.get("hidden_dim", 256)
    num_layers = hparams.get("num_layers", 3)
    num_blocks = hparams.get("num_blocks", 2)
    nhead = hparams.get("nhead", 4)

    if args.model == "XGB":
        model = XGBoostWrapper(
            n_estimators=hparams.get("n_estimators", 100),
            max_depth=hparams.get("max_depth", 6),
            learning_rate=hparams.get("learning_rate", 0.1),
            min_child_weight=hparams.get("min_child_weight", 1.0),
            subsample=hparams.get("subsample", 1.0),
            colsample_bylevel=hparams.get("colsample_bylevel", 1.0),
            colsample_bytree=hparams.get("colsample_bytree", 1.0),
            gamma=hparams.get("gamma", 0.0),
            reg_lambda=hparams.get("reg_lambda", 1.0),
            reg_alpha=hparams.get("reg_alpha", 0.0),
            random_state=seed,
            patience=patience,
        )
    elif args.model == "LGB":
        model = LightGBMWrapper(
            n_estimators=hparams.get("n_estimators", 100),
            num_leaves=hparams.get("num_leaves", 31),
            learning_rate=hparams.get("learning_rate", 0.1),
            min_child_samples=hparams.get("min_child_samples", 20),
            subsample=hparams.get("subsample", 1.0),
            colsample_bytree=hparams.get("colsample_bytree", 1.0),
            random_state=seed,
            patience=patience,
        )
    elif args.model == "TabNet":
        opt_params = dict(lr=lr)
        if "weight_decay" in hparams:
            opt_params["weight_decay"] = hparams["weight_decay"]
        model = TabNetWrapper(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=opt_params,
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
            seed=seed,
            n_d=hparams.get("n_d", 8),
            n_a=hparams.get("n_a", 8),
            n_steps=hparams.get("n_steps", 3),
            gamma=hparams.get("gamma", 1.3),
            lambda_sparse=hparams.get("lambda_sparse", 1e-3),
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
        )
    elif args.model == "SAINT":
        saint_hparams = dict(hparams)
        _input_dim = saint_hparams.pop("input_dim", 32)
        _n_heads = saint_hparams.pop("n_heads", 4)
        _n_blocks = saint_hparams.pop("n_blocks", 2)
        _dropout = saint_hparams.pop("dropout", 0.1)
        saint_hparams.pop("batch_size", None)
        model = WidedeepWrapper(
            model_type="SAINT",
            input_dim=_input_dim,
            n_heads=_n_heads,
            n_blocks=_n_blocks,
            dropout=_dropout,
            mlp_dropout=_dropout,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            efficient_attention=args.efficient_attention,
            **saint_hparams,
        )
    elif args.model == "TabTransformer":
        tt_hparams = dict(hparams)
        use_efficient = args.efficient_attention
        _input_dim = tt_hparams.pop("input_dim", 32)
        _n_heads = tt_hparams.pop("n_heads", 4)
        _n_blocks = tt_hparams.pop("n_blocks", 2)
        _dropout = tt_hparams.pop("dropout", 0.1)
        tt_hparams.pop("batch_size", None)
        model = WidedeepWrapper(
            model_type="TabTransformer",
            input_dim=_input_dim,
            n_heads=_n_heads,
            n_blocks=_n_blocks,
            dropout=_dropout,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            efficient_attention=use_efficient,
            **tt_hparams,
        )
    elif args.model == "FTTransformer":
        ft_hparams = dict(hparams)
        _input_dim = ft_hparams.pop("input_dim", 192)
        _n_heads = ft_hparams.pop("n_heads", 8)
        _n_blocks = ft_hparams.pop("n_blocks", 2)
        ft_hparams.pop("batch_size", None)
        model = WidedeepWrapper(
            model_type="FTTransformer",
            input_dim=_input_dim,
            n_heads=_n_heads,
            n_blocks=_n_blocks,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            efficient_attention=args.efficient_attention,
            **ft_hparams,
        )
    elif args.model == "DCN":
        dcn_hparams = dict(hparams)
        dcn_hparams["cross_num"] = dcn_hparams.pop("n_cross_layers", 2)
        _dnn_hidden_units = dcn_hparams.pop("dnn_hidden_units", (256, 128))
        _dropout = dcn_hparams.pop("hidden_dropout", dcn_hparams.pop("dropout", 0.1))
        weight_decay = dcn_hparams.pop("weight_decay", 0.0)
        dcn_hparams.pop("batch_size", None)
        dcn_hparams.pop("cross_dropout", None)
        dcn_hparams.pop("layer_size", None)
        dcn_hparams["l2_reg_dnn"] = weight_decay
        dcn_hparams["l2_reg_cross"] = weight_decay
        model = DeepCTRWrapper(
            model_type="DCN",
            dnn_hidden_units=_dnn_hidden_units,
            dnn_dropout=_dropout,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            **dcn_hparams,
        )
    elif args.model == "AutoInt":
        autoint_hparams = dict(hparams)
        _dropout = autoint_hparams.pop("dropout", 0.1)
        # deepctr_torch AutoInt does not support att_embedding_dim in this environment.
        autoint_hparams.pop("att_embedding_dim", None)
        autoint_hparams.pop("batch_size", None)
        model = DeepCTRWrapper(
            model_type="AutoInt",
            dnn_dropout=_dropout,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            **autoint_hparams,
        )
    elif args.model == "MLP":
        net = MLP(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
        )
        model = train_torch_model(
            net,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=hparams.get("weight_decay", 0.0),
            patience=patience,
        )
    elif args.model == "ResNet":
        net = ResNet(
            input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout
        )
        model = train_torch_model(
            net,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=hparams.get("weight_decay", 0.0),
            patience=patience,
        )
    elif args.model == "DANN":
        net = DANN(
            input_dim=input_dim,
            num_classes=2,
            num_domains=num_domains,
            hparams={
                **hparams,
                "lr": lr,
                "backbone": backbone,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "nhead": nhead,
            },
        )
        model = train_dann(
            net,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            X_target=X_target,
        )
    elif args.model == "DAN":
        net = DAN(
            input_dim=input_dim,
            num_classes=2,
            hparams={
                **hparams,
                "lr": lr,
                "backbone": backbone,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "nhead": nhead,
            },
        )
        model = train_dan(
            net,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            X_target=X_target,
        )
    elif args.model == "IRM":
        model = IRM(
            input_dim=input_dim, num_classes=2, hparams={"lr": lr, "backbone": backbone, **hparams}
        )
    elif args.model == "VREx":
        model = VREx(
            input_dim=input_dim, num_classes=2, hparams={"lr": lr, "backbone": backbone, **hparams}
        )
    elif args.model == "GroupDRO":
        model = GroupDRO(
            input_dim=input_dim,
            num_classes=2,
            hparams={"lr": lr, "num_domains": num_domains, "backbone": backbone, **hparams},
        )
    elif args.model == "MixStyle":
        model = MixStyle(
            input_dim=input_dim, num_classes=2, hparams={"lr": lr, "backbone": backbone, **hparams}
        )
    elif args.model == "ERM_DG":
        model = DG_ERM(
            input_dim=input_dim, num_classes=2, hparams={"lr": lr, "backbone": backbone, **hparams}
        )
    elif args.model == "MLDG":
        model = MLDG(
            input_dim=input_dim, num_classes=2, hparams={"lr": lr, "backbone": backbone, **hparams}
        )
    elif args.model == "MASF":
        model = MASF(
            input_dim=input_dim, num_classes=2, hparams={"lr": lr, "backbone": backbone, **hparams}
        )
    elif args.model == "Fish":
        model = Fish(
            input_dim=input_dim, num_classes=2, hparams={"lr": lr, "backbone": backbone, **hparams}
        )
    elif args.model == "CSD":
        model = CSD(
            input_dim=input_dim, num_classes=2, hparams={"lr": lr, "backbone": backbone, **hparams}
        )
    elif args.model == "SagNet":
        model = SagNet(
            input_dim=input_dim, num_classes=2, hparams={"lr": lr, "backbone": backbone, **hparams}
        )
    elif args.model == "ADDA":
        net = ADDA(
            input_dim=input_dim,
            num_classes=2,
            hparams={
                **hparams,
                "lr": lr,
                "backbone": backbone,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "nhead": nhead,
            },
        )
        model = train_adda(
            net,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            X_target=X_target,
        )
    elif args.model == "MCD":
        net = MCD(
            input_dim=input_dim,
            num_classes=2,
            hparams={
                **hparams,
                "lr": lr,
                "backbone": backbone,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "nhead": nhead,
            },
        )
        mcd_model = train_mcd(
            net,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            X_target=X_target,
        )
        model = MCDInferenceWrapper(mcd_model)
        attach_training_metadata(model, **dict(getattr(mcd_model, "_training_info", {})))
    elif args.model == "JAN":
        net = JAN(
            input_dim=input_dim,
            num_classes=2,
            hparams={
                **hparams,
                "lr": lr,
                "backbone": backbone,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "nhead": nhead,
            },
        )
        model = train_jan(
            net,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            X_target=X_target,
        )
    elif args.model == "SHOT":
        net = SHOT(
            input_dim=input_dim,
            num_classes=2,
            hparams={
                **hparams,
                "lr": lr,
                "backbone": backbone,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "nhead": nhead,
            },
        )
        model = train_shot(
            net,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            X_target=X_target,
        )
    elif args.model == "CBST":
        net = CBST(
            input_dim=input_dim,
            num_classes=2,
            hparams={
                **hparams,
                "lr": lr,
                "backbone": backbone,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "nhead": nhead,
            },
        )
        model = train_cbst(
            net,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            X_target=X_target,
        )
    elif args.model == "CDAN":
        net = CDAN(
            input_dim=input_dim,
            num_classes=2,
            num_domains=num_domains,
            hparams={
                **hparams,
                "lr": lr,
                "backbone": backbone,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "nhead": nhead,
            },
        )
        model = train_cdan(
            net,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            X_target=X_target,
        )
    elif args.model == "DeepCORAL":
        net = DeepCORAL(
            input_dim=input_dim,
            num_classes=2,
            hparams={
                **hparams,
                "lr": lr,
                "backbone": backbone,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "nhead": nhead,
            },
        )
        model = train_deepcoral(
            net,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            X_target=X_target,
        )
    elif args.model == "MCC":
        net = MCC(
            input_dim=input_dim,
            num_classes=2,
            hparams={
                **hparams,
                "lr": lr,
                "backbone": backbone,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "nhead": nhead,
            },
        )
        model = train_mcc(
            net,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            X_target=X_target,
        )
    elif args.model == "CGDM":
        if X_target is None:
            raise ValueError("CGDM requires X_target (use --uda).")
        net = CGDM(input_dim=input_dim, num_classes=num_classes)
        model = train_cgdm(
            net,
            X_train,
            y_train,
            X_target,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            weight_decay=hparams.get("weight_decay", 5e-4),
            num_k=hparams.get("num_k", 1),
        )

    if args.model in [
        "XGB",
        "LGB",
        "TabNet",
        "SAINT",
        "TabTransformer",
        "FTTransformer",
        "DCN",
        "AutoInt",
    ]:
        model.fit(X_train, y_train, X_val, y_val)
    elif args.model in [
        "IRM",
        "VREx",
        "GroupDRO",
        "MixStyle",
        "MLDG",
        "MASF",
        "Fish",
        "CSD",
        "SagNet",
    ]:
        model = train_dg_model(
            model,
            X_train,
            y_train,
            d_train,
            X_val,
            y_val,
            d_val,
            epochs=epochs,
            batch_size=batch_size,
            domains_per_batch=8,
            patience=patience,
        )

    logged_hparams = dict(hparams)
    logged_hparams["batch_size"] = batch_size

    attach_training_metadata(
        model,
        seed=seed,
        selected_lr=lr,
        selected_batch_size=batch_size,
        configured_default_batch_size=args.batch_size,
        configured_max_epochs=args.epochs,
        max_epochs=epochs,
        patience=patience,
        backbone=backbone,
        model_name=args.model,
        hparams=logged_hparams,
        early_stopping_enabled=bool(patience and patience > 0),
    )
    return model


def main():
    args = get_args()
    if args.model in ("TFTransformer", "TF-transformer"):
        args.model = "FTTransformer"
    if args.split_strategy == "temporal":
        if args.temporal_train_ratio <= 0 or args.temporal_val_ratio <= 0:
            raise ValueError("Temporal ratios must be > 0.")
        if args.temporal_train_ratio + args.temporal_val_ratio >= 1.0:
            raise ValueError("Temporal train_ratio + val_ratio must be < 1.0.")

    output_path = Path(args.output)
    progress_output_path = output_path.with_name(output_path.stem + "_progress.csv")

    logger.info(f"Loading {args.dataset} (Label: {args.label})...")

    if args.label == "stress_binary":
        if args.dataset == "D-1":
            dataset_path = os.path.join(BASE_DATA_DIR, "stress_binary_personal-full_D-1.pkl")
        elif args.dataset == "D-2":
            dataset_path = os.path.join(BASE_DATA_DIR, "stress_binary_personal-full_D-2.pkl")
        elif args.dataset == "D-3":
            dataset_path = os.path.join(BASE_DATA_DIR, "stress_binary_personal-full_D-3.pkl")
        else:
            raise ValueError("Unknown dataset")
    elif args.dataset == "D-1":
        dataset_path = os.path.join(BASE_DATA_DIR, f"{args.label}_personal-full_D#2.pkl")
    elif args.dataset == "D-2":
        dataset_path = os.path.join(BASE_DATA_DIR, f"{args.label}_personal-full_D#3.pkl")
    elif args.dataset == "D-3":
        dataset_path = os.path.join(BASE_DATA_DIR, f"{args.label}_personal-full.pkl")
    else:
        raise ValueError("Unknown dataset")

    ds = BenchmarkDataset(args.dataset, dataset_path)

    split_seed = 42
    labels = ds.y
    groups = ds.users
    X_raw = ds.X.copy()

    fold_splits = []
    if args.split_strategy == "group_kfold":
        group_folds = 5
        splitter = StratifiedGroupKFold(n_splits=group_folds, shuffle=True, random_state=split_seed)
        for fold_id, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros_like(labels), labels, groups)
        ):
            train_idx, val_idx = make_groupwise_val_split(
                train_idx, labels, groups, seed=split_seed + fold_id
            )
            fold_splits.append((fold_id, train_idx, val_idx, test_idx))
    else:
        group_folds = 1
        train_idx, val_idx, test_idx, reason_counts = make_temporal_global_split(
            labels=labels,
            users=groups,
            timestamps=ds.timestamps,
            train_ratio=args.temporal_train_ratio,
            val_ratio=args.temporal_val_ratio,
            drop_days=args.temporal_drop_days,
        )
        logger.info(
            "Temporal split stats: "
            f"kept_users={reason_counts['kept_users']}, "
            f"dropped_insufficient_samples={reason_counts['dropped_insufficient_samples']}, "
            f"dropped_insufficient_label_diversity={reason_counts['dropped_insufficient_label_diversity']}"
        )
        fold_splits.append((0, train_idx, val_idx, test_idx))

    if args.split_strategy == "temporal" and args.rebin_personal != "none":
        if len(fold_splits) != 1:
            raise NotImplementedError(
                "rebin_personal currently assumes a single fold (temporal Tier A). "
                "Multi-fold rebinarization needs per-fold y snapshots before normalize_features."
            )
        agg = "median" if args.rebin_personal == "train_median" else "mean"
        raw_label = ds.get_raw_label_aligned(args.label)
        old_y_overall = np.unique(ds.y, return_counts=True)
        rebuilt_splits = []
        for fold_id, train_idx, val_idx, test_idx in fold_splits:
            y_new, thresholds = rebinarize_personal_train_only(
                ds.users,
                raw_label,
                train_idx,
                agg=agg,
            )
            ds.y = y_new
            new_train, new_val, new_test, diversity_stats = filter_split_for_label_diversity(
                ds.y,
                ds.users,
                train_idx,
                val_idx,
                test_idx,
            )
            logger.info(
                f"[Tier A rebin] label={args.label} fold={fold_id} agg={agg} "
                f"old_class_balance={dict(zip(old_y_overall[0].tolist(), old_y_overall[1].tolist()))} "
                f"new_class_balance={dict(zip(*[a.tolist() for a in np.unique(ds.y, return_counts=True)]))} "
                f"users_kept={diversity_stats['kept_users']} "
                f"users_dropped_post_rebin={diversity_stats['dropped_no_diversity_after_rebin']} "
                f"sizes: train {len(train_idx)}->{len(new_train)} "
                f"val {len(val_idx)}->{len(new_val)} test {len(test_idx)}->{len(new_test)}"
            )
            logger.info(f"[Tier A rebin] per-user thresholds (fold {fold_id}): {thresholds}")
            if len(new_train) == 0 or len(new_val) == 0 or len(new_test) == 0:
                raise ValueError(
                    f"Rebinarization left fold {fold_id} empty in train/val/test. "
                    "Try --rebin_personal none or relax temporal split parameters."
                )
            rebuilt_splits.append((fold_id, new_train, new_val, new_test))
        fold_splits = rebuilt_splits
        labels = ds.y

    if args.max_folds is not None:
        fold_splits = fold_splits[: args.max_folds]

    le = LabelEncoder()
    user_domain_ids = le.fit_transform(ds.users)
    num_domains_all = len(le.classes_)

    fold_data = []
    for fold_id, train_idx, val_idx, test_idx in fold_splits:
        ds.X = X_raw.copy()
        ds.normalize_features(train_idx, val_idx, test_idx)
        fold_data.append(
            {
                "fold_id": fold_id,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "X_train": ds.X[train_idx],
                "y_train": ds.y[train_idx],
                "X_val": ds.X[val_idx],
                "y_val": ds.y[val_idx],
                "X_test": ds.X[test_idx],
                "y_test": ds.y[test_idx],
            }
        )

    DG_MODELS = [
        "DANN",
        "CDAN",
        "DAN",
        "DeepCORAL",
        "MCC",
        "CGDM",
        "IRM",
        "VREx",
        "GroupDRO",
        "MixStyle",
        "ERM_DG",
        "MLDG",
        "MASF",
        "ADDA",
        "MCD",
        "JAN",
        "SHOT",
        "CBST",
        "Fish",
        "CSD",
        "SagNet",
    ]
    DA_MODELS = [
        "DANN",
        "CDAN",
        "DAN",
        "DeepCORAL",
        "MCC",
        "ADDA",
        "MCD",
        "JAN",
        "SHOT",
        "CBST",
        "CGDM",
    ]

    def _prepare_domain_info(entry):
        if args.model not in DG_MODELS:
            return None, None, 0, None
        if args.uda:
            d_train = np.zeros(len(entry["y_train"]), dtype=int)
            d_val = np.zeros(len(entry["y_val"]), dtype=int)
            return d_train, d_val, 2, entry["X_test"]
        d_train = user_domain_ids[entry["train_idx"]]
        d_val = user_domain_ids[entry["val_idx"]]
        return d_train, d_val, num_domains_all, None

    best_hparams = {}

    def _run_hpo(folds_for_hpo, *, label):
        logger.info(f"Starting {label} with {args.hpo_trials} trials...")

        def objective(trial):
            hparams = get_hparams(args.model, args.dataset, backbone=args.backbone)
            trial_params = {}
            for k, v in hparams.items():
                if callable(v):
                    trial_params[k] = v(trial)
                else:
                    trial_params[k] = v
            trial.set_user_attr("resolved_hparams", dict(trial_params))

            scores = []
            for entry in folds_for_hpo:
                d_train, d_val, num_domains, X_target = _prepare_domain_info(entry)
                X_val_eval = entry["X_val"]
                y_val_eval = entry["y_val"]
                model = None
                try:
                    model = train_model(
                        args,
                        entry["X_train"],
                        entry["y_train"],
                        d_train,
                        X_val_eval,
                        y_val_eval,
                        d_val,
                        entry["X_train"].shape[1],
                        2,
                        num_domains,
                        trial_params,
                        seed=42,
                        patience=args.patience,
                        X_target=X_target,
                    )
                    val_metrics = evaluate_model(model, X_val_eval, y_val_eval)
                    scores.append(val_metrics["AUROC"])
                except Exception as e:
                    logger.info(f"HPO Trial failed on fold {entry['fold_id'] + 1}: {e}")
                    return 0.0
                finally:
                    model = None
                    release_torch_memory()
            return float(np.mean(scores)) if scores else 0.0

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=args.hpo_trials)
        best_params = dict(study.best_trial.user_attrs.get("resolved_hparams", {})) or dict(
            study.best_params
        )
        logger.info("Best HPO params:", best_params)
        return best_params, study

    hpo_study = None
    if args.hpo_trials > 0 and args.hpo_mode in ("fold1", "cv"):
        folds_for_hpo = fold_data if args.hpo_mode == "cv" else [fold_data[0]]
        best_hparams, hpo_study = _run_hpo(
            folds_for_hpo, label="CV-HPO" if args.hpo_mode == "cv" else "Fold-1 HPO"
        )

    seeds = parse_seeds(args.seeds)
    records_dir = str(Path(args.output).parent / "records")
    configured_max_epochs = args.epochs_override if args.epochs_override else args.epochs

    METRIC_KEYS = [
        "Train_Accuracy",
        "Train_AUROC",
        "Train_F1",
        "Train_Precision",
        "Train_Recall",
        "Val_Accuracy",
        "Val_AUROC",
        "Val_F1",
        "Val_Precision",
        "Val_Recall",
        "Test_Accuracy",
        "Test_F1",
        "Test_AUROC",
        "Test_Precision",
        "Test_Recall",
    ]
    all_fold_results = []

    for entry in fold_data:
        fold_id = entry["fold_id"]
        logger.info(f"\n=== Fold {fold_id + 1}/{group_folds} ===")

        fold_hparams = best_hparams
        fold_study = hpo_study
        if args.hpo_trials > 0 and args.hpo_mode == "nested":
            fold_hparams, fold_study = _run_hpo([entry], label=f"Nested HPO (Fold {fold_id + 1})")

        d_train, d_val, num_domains, X_target = _prepare_domain_info(entry)
        logger.info(
            f"Data Splits: Train {entry['X_train'].shape}, Val {entry['X_val'].shape}, Test {entry['X_test'].shape}"
        )

        for seed in seeds:
            logger.info(f"\n--- Fold {fold_id + 1} | Seed {seed} ---")

            benchmark_logger = BenchmarkLogger(output_dir=records_dir, benchmark_type="cross_user")

            clip_val = float(max(10.0, np.percentile(np.abs(entry["X_train"].reshape(-1)), 99.9)))
            benchmark_logger.set_setting(
                dataset=args.dataset,
                label=args.label,
                model=args.model,
                backbone=args.backbone,
                seed=seed,
                fold_id=fold_id,
                n_folds=group_folds,
                split_strategy=args.split_strategy,
                hpo_trials=args.hpo_trials,
                hpo_mode=args.hpo_mode,
                max_epochs=args.epochs,
                patience=args.patience,
            )
            benchmark_logger.record_policy(
                seeds=seeds,
                planned_hpo_trials=args.hpo_trials,
                hpo_mode=args.hpo_mode,
                max_epochs=configured_max_epochs,
                default_batch_size=args.batch_size,
                patience=args.patience,
            )
            benchmark_logger.set_preprocessing(clip_value=clip_val)
            benchmark_logger.set_split_stats(
                entry["y_train"],
                ds.users[entry["train_idx"]],
                entry["y_val"],
                ds.users[entry["val_idx"]],
                entry["y_test"],
                ds.users[entry["test_idx"]],
            )
            if fold_study is not None:
                benchmark_logger.record_hpo(fold_study, fold_hparams)
            else:
                benchmark_logger.record_hpo_no_study(fold_hparams)

            X_val_train = entry["X_val"]
            y_val_train = entry["y_val"]

            with benchmark_logger.time_train():
                model = train_model(
                    args,
                    entry["X_train"],
                    entry["y_train"],
                    d_train,
                    X_val_train,
                    y_val_train,
                    d_val,
                    entry["X_train"].shape[1],
                    2,
                    num_domains,
                    hparams=fold_hparams,
                    seed=seed,
                    patience=args.patience,
                    X_target=X_target,
                )

            benchmark_logger.record_training(model)
            benchmark_logger.record_model_stats(model, input_dim=entry["X_train"].shape[1])
            selected_batch_size = benchmark_logger._rec["training"].get("batch_size")

            with benchmark_logger.time_eval():
                train_m = evaluate_extended(
                    model, entry["X_train"], entry["y_train"], batch_size=selected_batch_size
                )
                val_m = evaluate_extended(
                    model, entry["X_val"], entry["y_val"], batch_size=selected_batch_size
                )
                test_m = evaluate_extended(
                    model, entry["X_test"], entry["y_test"], batch_size=selected_batch_size
                )

            benchmark_logger.record_metrics(train_m, val_m, test_m)
            benchmark_logger.record_inference_benchmark(
                model, entry["X_test"], batch_size=selected_batch_size, split_name="test"
            )
            record = benchmark_logger.finalize()
            rt = record["runtime"]
            tr = record["training"]
            ss = record["split_stats"]
            hw = record["hardware"]
            budget = record["compute_budget"]
            model_stats = record["model_stats"]
            infer = record["inference_benchmark"].get("test", {})
            sustain = record["sustainability"]

            logger.info(
                f"  Train AUROC={train_m['auroc']:.4f}  Val AUROC={val_m['auroc']:.4f}  Test AUROC={test_m['auroc']:.4f}  wall={rt['total_wall_s']}s"
            )

            fold_result = {
                "Train_Accuracy": train_m["accuracy"],
                "Train_AUROC": train_m["auroc"],
                "Train_F1": train_m["f1"],
                "Train_Precision": train_m["precision"],
                "Train_Recall": train_m["recall"],
                "Val_Accuracy": val_m["accuracy"],
                "Val_AUROC": val_m["auroc"],
                "Val_F1": val_m["f1"],
                "Val_Precision": val_m["precision"],
                "Val_Recall": val_m["recall"],
                "Test_Accuracy": test_m["accuracy"],
                "Test_F1": test_m["f1"],
                "Test_AUROC": test_m["auroc"],
                "Test_Precision": test_m["precision"],
                "Test_Recall": test_m["recall"],
            }
            all_fold_results.append(fold_result)

            progress_row = {
                "Dataset": args.dataset,
                "Label": args.label,
                "Model": args.model,
                "Backbone": args.backbone,
                "Seed": seed,
                "Fold": fold_id + 1,
                "Phase": "final",
                "Trial": "",
                "Train_Accuracy": train_m["accuracy"],
                "Train_AUROC": train_m["auroc"],
                "Train_F1": train_m["f1"],
                "Train_Precision": train_m["precision"],
                "Train_Recall": train_m["recall"],
                "Val_Accuracy": val_m["accuracy"],
                "Val_AUROC": val_m["auroc"],
                "Val_F1": val_m["f1"],
                "Val_Precision": val_m["precision"],
                "Val_Recall": val_m["recall"],
                "Test_Accuracy": test_m["accuracy"],
                "Test_F1": test_m["f1"],
                "Test_AUROC": test_m["auroc"],
                "Test_Precision": test_m["precision"],
                "Test_Recall": test_m["recall"],
                "Train_Samples": ss["train"]["n_samples"],
                "Val_Samples": ss["val"]["n_samples"],
                "Test_Samples": ss["test"]["n_samples"],
                "Train_Users": ss["train"]["n_users"],
                "Val_Users": ss["val"]["n_users"],
                "Test_Users": ss["test"]["n_users"],
                "Train_PosRatio": ss["train"]["positive_ratio"],
                "Val_PosRatio": ss["val"]["positive_ratio"],
                "Test_PosRatio": ss["test"]["positive_ratio"],
                "Best_Epoch": tr.get("best_epoch"),
                "Early_Stopped": tr.get("early_stopped"),
                "Seed_Count": budget.get("seed_count"),
                "HPO_Best_AUROC": record["hpo"].get("best_value"),
                "HPO_Planned_Trials": budget.get("planned_hpo_trials"),
                "HPO_Completed_Trials": budget.get("completed_hpo_trials"),
                "Configured_Max_Epochs": budget.get("max_epochs_per_run"),
                "Configured_Default_Batch_Size": budget.get("default_batch_size"),
                "Selected_Batch_Size": tr.get("batch_size"),
                "Selected_LR": tr.get("lr"),
                "Total_Wall_S": rt.get("total_wall_s"),
                "HPO_Wall_S": rt.get("hpo_wall_s"),
                "Train_Wall_S": rt.get("train_wall_s"),
                "Eval_Wall_S": rt.get("eval_wall_s"),
                "Train_GPU_Hours": rt.get("train_gpu_hours"),
                "Eval_GPU_Hours": rt.get("eval_gpu_hours"),
                "Total_GPU_Hours": rt.get("total_gpu_hours"),
                "Device_Name": hw.get("device_name"),
                "GPU_Model": hw.get("gpu_model"),
                "GPU_Count": hw.get("gpu_count"),
                "GPU_Total_VRAM_GB": hw.get("gpu_total_vram_gb"),
                "CPU_Model": hw.get("cpu_model"),
                "RAM_GB": hw.get("ram_gb"),
                "Peak_GPU_MB": rt.get("peak_gpu_memory_mb"),
                "Peak_CPU_MB": rt.get("peak_cpu_memory_mb"),
                "Param_Count": model_stats.get("parameter_count"),
                "Trainable_Param_Count": model_stats.get("trainable_parameter_count"),
                "Artifact_Size_MB": model_stats.get("artifact_size_mb"),
                "Inference_Batch_Size": infer.get("batch_size"),
                "Inference_Latency_MS": infer.get("per_batch_latency_ms"),
                "Inference_Throughput_SPS": infer.get("throughput_samples_per_s"),
                "FLOPs": model_stats.get("flops"),
                "MACs": model_stats.get("macs"),
                "Energy_KWh": sustain.get("energy_kwh"),
                "Carbon_KgCO2eq": sustain.get("carbon_kg_co2eq"),
                "Hparams_JSON": json.dumps(fold_hparams, default=str),
                "Experiment_ID": record["experiment_id"],
            }
            append_row(progress_output_path, progress_row, PROGRESS_COLUMNS)
            model = None
            release_torch_memory()

    if all_fold_results:
        summary = {
            "Dataset": args.dataset,
            "Label": args.label,
            "Model": args.model,
            "Backbone": args.backbone,
            "N_Folds": len(all_fold_results),
            "Seed_Count": len(seeds),
            "N_Runs": len(all_fold_results),
        }
        for key in METRIC_KEYS:
            vals = [r[key] for r in all_fold_results]
            summary[f"{key}_Mean"] = round(float(np.mean(vals)), 6)
            summary[f"{key}_Std"] = round(float(np.std(vals)), 6)
        if os.path.dirname(args.output):
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        header = not os.path.exists(args.output)
        pd.DataFrame([summary]).to_csv(args.output, mode="a", header=header, index=False)
        logger.info(f"\n=== Summary (Mean ± Std over {len(all_fold_results)} folds) ===")
        for key in ["Test_AUROC", "Test_F1", "Test_Accuracy"]:
            logger.info(f"  {key}: {summary[f'{key}_Mean']:.4f} ± {summary[f'{key}_Std']:.4f}")
        logger.info(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
