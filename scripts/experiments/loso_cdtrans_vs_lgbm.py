#!/usr/bin/env python3
"""Evaluation script comparing Optuna-tuned LightGBM and CDTrans under
LOSO or stratified K-fold protocols."""

import argparse
import copy
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


###############################################################################
# CDTrans model components (adapted locally)
###############################################################################


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_embedding(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor,
                  key: torch.Tensor,
                  value: torch.Tensor,
                  dropout: nn.Dropout) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        scores = scores.softmax(dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ value, scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch = q.size(0)
        q_proj = self.w_q(q).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        x, _ = MultiHeadAttention.attention(q_proj, k_proj, v_proj, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class CDTransLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.residual1 = ResidualConnection(dropout)
        self.residual2 = ResidualConnection(dropout)
        self.ff = FeedForwardBlock(d_model, d_ff=4 * d_model, dropout=dropout)

    def forward(self,
                h_s_prev: torch.Tensor,
                h_t_prev: torch.Tensor,
                h_st_prev: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_s = self.residual1(h_s_prev, lambda x: self.mha(x, x, x))
        h_s = self.residual2(h_s, self.ff)
        h_t = self.residual1(h_t_prev, lambda x: self.mha(x, x, x))
        h_t = self.residual2(h_t, self.ff)
        h_st = self.residual1(h_s_prev, lambda x: self.mha(x, h_t_prev, h_t_prev))
        h_st = self.residual2(h_st, self.ff)
        if h_st_prev is not None:
            h_st = h_st + h_st_prev
        return h_s, h_t, h_st

    def train_source(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual1(x, lambda x: self.mha(x, x, x))
        x = self.residual2(x, self.ff)
        return x


class CDTransEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([CDTransLayer(d_model, n_heads, dropout) for _ in range(num_layers)])

    def forward(self, h_s: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_st = None
        for layer in self.layers:
            h_s, h_t, h_st = layer(h_s, h_t, h_st)
        return h_s, h_t, h_st

    def train_source(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer.train_source(x)
        return x


class CDTransModel(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 num_classes: int = 2,
                 f_dim: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = f_dim
        self.src_input_embedding = nn.ModuleList([InputEmbedding(d_model) for _ in range(f_dim)])
        self.src_feature_embedding = nn.Embedding(f_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.encoder = CDTransEncoder(d_model, n_heads, n_layers, dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        embeddings = []
        for i in range(self.seq_len):
            feat = x[:, i].unsqueeze(-1)
            emb = self.src_input_embedding[i](feat)
            emb = emb + self.src_feature_embedding(torch.tensor(i, device=x.device))
            embeddings.append(emb)
        h = torch.stack(embeddings, dim=1)
        cls = self.cls_token.expand(batch, -1, -1)
        return torch.cat([cls, h], dim=1)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        h_s = self.embed(x_s)
        h_t = self.embed(x_t)
        H_s, H_t, H_st = self.encoder(h_s, h_t)
        feat_s = H_s[:, 0, :]
        feat_t = H_t[:, 0, :]
        feat_st = H_st[:, 0, :]
        logit_s = self.classifier(feat_s)
        logit_t = self.classifier(feat_t)
        logit_st = self.classifier(feat_st)
        return logit_s, logit_t, logit_st, H_s, H_t, H_st

    def train_source(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(x)
        H = self.encoder.train_source(h)
        feat = H[:, 0, :]
        logits = self.classifier(feat)
        return logits, feat


class VanillaTransformerModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 num_classes: int,
                 f_dim: int,
                 dropout: float):
        super().__init__()
        self.seq_len = f_dim
        self.input_embedding = nn.ModuleList([InputEmbedding(d_model) for _ in range(f_dim)])
        self.feature_embedding = nn.Embedding(f_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.encoder = CDTransEncoder(d_model, n_heads, n_layers, dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        embeddings = []
        for i in range(self.seq_len):
            feat = x[:, i].unsqueeze(-1)
            emb = self.input_embedding[i](feat)
            emb = emb + self.feature_embedding(torch.tensor(i, device=x.device))
            embeddings.append(emb)
        h = torch.stack(embeddings, dim=1)
        cls = self.cls_token.expand(batch, -1, -1)
        return torch.cat([cls, h], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(x)
        encoded = self.encoder.train_source(h)
        feat = encoded[:, 0, :]
        logits = self.classifier(feat)
        return logits, feat


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, 'rb') as f:
        df, labels, users, *_ = pickle.load(f)
    X = df.values.astype(np.float32) if hasattr(df, 'values') else np.asarray(df, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    users = np.asarray(users)
    return X, y, users


def metric_dict(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auroc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'prauc': average_precision_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def print_metrics(tag: str, metrics: Dict[str, float]) -> None:
    print(f"{tag}: Acc={metrics['acc']:.3f}, AUROC={metrics['auroc']:.3f}, PRAUC={metrics['prauc']:.3f}, F1={metrics['f1']:.3f}")


class EarlyStopper:
    def __init__(self, patience: int, mode: str = 'max', min_delta: float = 0.0):
        self.patience = max(0, patience)
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.bad_epochs = 0

    @staticmethod
    def _is_nan(value: Optional[float]) -> bool:
        if value is None:
            return False
        try:
            return bool(np.isnan(value))
        except TypeError:
            return False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None or self._is_nan(self.best_score):
            return True
        if self.mode == 'max':
            return score > (self.best_score + self.min_delta)
        return score < (self.best_score - self.min_delta)

    def step(self, score: float, model: nn.Module) -> Tuple[bool, bool]:
        if self._is_nan(score):
            self.bad_epochs += 1
            stop = self.bad_epochs >= self.patience
            return stop, False

        if self.best_score is None or self._is_nan(self.best_score) or self._is_improvement(score):
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
            return False, True

        self.bad_epochs += 1
        stop = self.bad_epochs >= self.patience
        return stop, False

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def get_monitor_value(metrics: Dict[str, float], metric_name: str, fallback: str = 'acc') -> float:
    primary = metrics.get(metric_name)
    if primary is None or EarlyStopper._is_nan(primary):
        primary = metrics.get(fallback)
    if primary is None or EarlyStopper._is_nan(primary):
        raise ValueError(f"Unable to retrieve monitoring metric '{metric_name}' (fallback '{fallback}')")
    return float(primary)


def round_metric(value: float, digits: int = 3) -> float:
    if EarlyStopper._is_nan(value):
        return float('nan')
    return round(float(value), digits)


def update_best_metrics(best: Dict[str, float], current: Dict[str, float]) -> None:
    for key, value in current.items():
        if EarlyStopper._is_nan(value):
            continue
        if key not in best or value > best[key]:
            best[key] = value


RESULT_COLUMNS = [
    'split',
    'auroc_lgbm',
    'auroc_vanilla_at_val', 'auroc_vanilla_best',
    'auroc_cdtrans_pretrain', 'auroc_cdtrans_at_val', 'auroc_cdtrans_best',
    'prauc_lgbm',
    'prauc_vanilla_at_val', 'prauc_vanilla_best',
    'prauc_cdtrans_pretrain', 'prauc_cdtrans_at_val', 'prauc_cdtrans_best'
]


def build_results_dataframe(rows: List[Dict[str, Union[float, str]]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.DataFrame(rows)
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[RESULT_COLUMNS]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].round(3)
    return df


def persist_results(rows: List[Dict[str, Union[float, str]]], output_csv: str) -> None:
    df = build_results_dataframe(rows)
    df.to_csv(output_csv, index=False)


def fmt_metric_str(value: float) -> str:
    rounded = round_metric(value)
    return f"{rounded:.3f}" if not math.isnan(rounded) else "nan"


def tune_lgbm(X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              n_trials: int,
              seed: int,
              num_threads: int) -> lgb.Booster:
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_pre_filter': False,
            'verbose': -1,
            'seed': seed,
            'num_threads': num_threads
        }
        callbacks = [
            lgb.early_stopping(50, verbose=False)
        ]
        model = lgb.train(params,
                          dtrain,
                          num_boost_round=500,
                          valid_sets=[dval],
                          valid_names=['val'],
                          callbacks=callbacks)
        prob = model.predict(X_val, num_iteration=model.best_iteration)
        return roc_auc_score(y_val, prob) if len(np.unique(y_val)) > 1 else 0.5

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
        'seed': seed,
        'feature_pre_filter': False,
        'num_threads': num_threads
    })
    callbacks = [
        lgb.early_stopping(50, verbose=True),
        lgb.log_evaluation(50)
    ]
    booster = lgb.train(best_params,
                        dtrain,
                        num_boost_round=500,
                        valid_sets=[dval],
                        valid_names=['val'],
                        callbacks=callbacks)
    return booster


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y_true = torch.from_numpy(y.astype(np.int64))
        self.pseudo: Optional[torch.Tensor] = None
        self.use_pseudo = False

    def __getitem__(self, idx: int):
        if self.use_pseudo and self.pseudo is not None:
            target = self.pseudo[idx]
        else:
            target = self.y_true[idx]
        return self.X[idx], target, idx

    def __len__(self) -> int:
        return len(self.X)


def update_pseudo(model: CDTransModel, loader: DataLoader, device: torch.device) -> torch.Tensor:
    dataset = loader.dataset
    prev_flag = getattr(dataset, 'use_pseudo', False)
    if hasattr(dataset, 'use_pseudo'):
        dataset.use_pseudo = False

    model.eval()
    feats, probs, indices = [], [], []
    with torch.no_grad():
        for batch_x, _, batch_idx in loader:
            batch_x = batch_x.to(device)
            logits, feat = model.train_source(batch_x)
            probs.append(F.softmax(logits, dim=1).cpu())
            feats.append(feat.cpu())
            indices.append(batch_idx)
    probs = torch.cat(probs, dim=0)
    feats = torch.cat(feats, dim=0)
    indices = torch.cat(indices, dim=0)

    num_classes = probs.size(1)
    centers = []
    for c in range(num_classes):
        weight = probs[:, c].unsqueeze(1)
        centers.append((weight * feats).sum(dim=0) / (weight.sum() + 1e-6))
    centers = torch.stack(centers)

    init_labels = torch.argmin(torch.cdist(feats, centers), dim=1)
    new_centers = []
    for c in range(num_classes):
        mask = init_labels == c
        new_centers.append(feats[mask].mean(dim=0) if mask.any() else centers[c])
    new_centers = torch.stack(new_centers)
    refined = torch.argmin(torch.cdist(feats, new_centers), dim=1)

    pseudo = torch.zeros_like(indices, dtype=torch.long)
    pseudo[indices] = refined

    if hasattr(dataset, 'use_pseudo'):
        dataset.use_pseudo = prev_flag

    return pseudo


def create_pairs(model: CDTransModel,
                 src_ds: TabularDataset,
                 tgt_ds: TabularDataset,
                 batch_size: int,
                 device: torch.device,
                 num_classes: int) -> DataLoader:
    def extract(ds: TabularDataset) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        feats, labels = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                logits, feat = model.train_source(x)
                feats.append(feat.cpu().numpy())
                labels.append(y.numpy())
        return np.concatenate(feats), np.concatenate(labels)

    prev_src_flag = getattr(src_ds, 'use_pseudo', False)
    prev_tgt_flag = getattr(tgt_ds, 'use_pseudo', False)
    src_ds.use_pseudo = False
    tgt_ds.use_pseudo = True

    src_feats, src_labels = extract(src_ds)
    tgt_feats, tgt_labels = extract(tgt_ds)

    src_ds.use_pseudo = prev_src_flag
    tgt_ds.use_pseudo = prev_tgt_flag

    pairs = set()
    for c in range(num_classes):
        src_idx = np.where(src_labels == c)[0]
        tgt_idx = np.where(tgt_labels == c)[0]
        if len(src_idx) == 0 or len(tgt_idx) == 0:
            continue
        for s in src_idx:
            dists = np.linalg.norm(tgt_feats[tgt_idx] - src_feats[s], axis=1)
            pairs.add((int(s), int(tgt_idx[np.argmin(dists)])))
        for t in tgt_idx:
            dists = np.linalg.norm(src_feats[src_idx] - tgt_feats[t], axis=1)
            pairs.add((int(src_idx[np.argmin(dists)]), int(t)))

    if not pairs:
        raise ValueError("No matched pairs for CDTrans adaptation")

    class PairDataset(Dataset):
        def __init__(self, src: TabularDataset, tgt: TabularDataset, pair_list: List[Tuple[int, int]]):
            self.src = src
            self.tgt = tgt
            self.pairs = pair_list

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int):
            s_idx, t_idx = self.pairs[idx]
            return self.src[s_idx], self.tgt[t_idx]

    return DataLoader(PairDataset(src_ds, tgt_ds, list(pairs)), batch_size=batch_size,
                      shuffle=True, drop_last=True)


def eval_cdtrans(model: CDTransModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    dataset = loader.dataset
    prev_flag = getattr(dataset, 'use_pseudo', False)
    if hasattr(dataset, 'use_pseudo'):
        dataset.use_pseudo = False

    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits, _ = model.train_source(x)
            prob = torch.sigmoid(logits[:, 1]) if logits.size(1) == 2 else F.softmax(logits, dim=1)[:, 1]
            y_prob.append(prob.cpu().numpy())
            y_true.append(y.numpy())

    if hasattr(dataset, 'use_pseudo'):
        dataset.use_pseudo = prev_flag

    return metric_dict(np.concatenate(y_true), np.concatenate(y_prob))


def eval_transformer(model: VanillaTransformerModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits, _ = model(x)
            if logits.size(1) == 2:
                prob = torch.softmax(logits, dim=1)[:, 1]
            else:
                prob = torch.sigmoid(logits[:, 0])
            y_prob.append(prob.cpu().numpy())
            y_true.append(y.numpy())
    return metric_dict(np.concatenate(y_true), np.concatenate(y_prob))




def evaluate_split(split_label: str,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   args: argparse.Namespace,
                   device: torch.device) -> Dict[str, Union[float, str]]:
    print("=" * 80)
    print(f"Split: {split_label}")

    row: Dict[str, Union[float, str]] = {
        'split': split_label,
        'auroc_lgbm': float('nan'),
        'prauc_lgbm': float('nan'),
        'auroc_vanilla_at_val': float('nan'),
        'auroc_vanilla_best': float('nan'),
        'prauc_vanilla_at_val': float('nan'),
        'prauc_vanilla_best': float('nan'),
        'auroc_cdtrans_pretrain': float('nan'),
        'auroc_cdtrans_at_val': float('nan'),
        'auroc_cdtrans_best': float('nan'),
        'prauc_cdtrans_pretrain': float('nan'),
        'prauc_cdtrans_at_val': float('nan'),
        'prauc_cdtrans_best': float('nan')
    }

    # LightGBM
    print()
    print("[LightGBM] Optuna tuning...")
    booster = tune_lgbm(X_train, y_train, X_val, y_val, args.lgbm_trials, args.seed, args.lgbm_num_threads)
    best_iter = booster.best_iteration or booster.current_iteration() or booster.num_trees()
    train_metrics = metric_dict(y_train, booster.predict(X_train, num_iteration=best_iter))
    val_metrics = metric_dict(y_val, booster.predict(X_val, num_iteration=best_iter))
    test_metrics = metric_dict(y_test, booster.predict(X_test, num_iteration=best_iter))
    print_metrics("[LGBM] Train", train_metrics)
    print_metrics("[LGBM] Val", val_metrics)
    print_metrics("[LGBM] Test", test_metrics)

    row['auroc_lgbm'] = round_metric(test_metrics['auroc'])
    row['prauc_lgbm'] = round_metric(test_metrics['prauc'])

    # Prepare datasets for neural models
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    test_ds = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    final_metrics_vanilla: Optional[Dict[str, float]] = None
    best_val_metrics_vanilla: Optional[Dict[str, float]] = None
    best_test_metrics_vanilla_overall: Dict[str, float] = {}

    if args.run_vanilla_transformer:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print()
        print("[VanillaTransformer] Training...")
        vanilla_model = VanillaTransformerModel(d_model=args.d_model,
                                                n_heads=args.n_heads,
                                                n_layers=args.n_layers,
                                                num_classes=2,
                                                f_dim=X_train.shape[1],
                                                dropout=args.dropout).to(device)
        vanilla_optimizer = optim.Adam(vanilla_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        vanilla_criterion = nn.CrossEntropyLoss()
        vanilla_monitor_metric = args.vanilla_transformer_metric.strip().lower()
        vanilla_monitor_mode = 'min' if vanilla_monitor_metric.endswith('loss') else 'max'
        vanilla_stopper = EarlyStopper(args.vanilla_transformer_patience,
                                       mode=vanilla_monitor_mode,
                                       min_delta=args.vanilla_transformer_min_delta) if args.vanilla_transformer_patience >= 0 else None
        vanilla_best_epoch = 0
        vanilla_stopped_early = False
        vanilla_best_score: Optional[float] = None
        vanilla_best_state: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(1, args.vanilla_transformer_epochs + 1):
            vanilla_model.train()
            running_loss = 0.0
            for x_batch, y_batch, _ in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits, _ = vanilla_model(x_batch)
                loss = vanilla_criterion(logits, y_batch)
                vanilla_optimizer.zero_grad()
                loss.backward()
                vanilla_optimizer.step()
                running_loss += loss.item() * x_batch.size(0)
            loss_epoch = running_loss / len(train_loader.dataset)
            val_metrics_vanilla_epoch = eval_transformer(vanilla_model, val_loader, device)
            test_metrics_vanilla_epoch = eval_transformer(vanilla_model, test_loader, device)
            print(f"[VanillaTransformer] Epoch {epoch}/{args.vanilla_transformer_epochs} Loss={loss_epoch:.3f}")
            print_metrics("    Val", val_metrics_vanilla_epoch)
            update_best_metrics(best_test_metrics_vanilla_overall,
                                {
                                    'auroc': test_metrics_vanilla_epoch['auroc'],
                                    'prauc': test_metrics_vanilla_epoch['prauc']
                                })

            if vanilla_stopper:
                try:
                    monitor_value = get_monitor_value(val_metrics_vanilla_epoch, vanilla_monitor_metric)
                except ValueError as exc:
                    print(f"[VanillaTransformer] Early stopping check skipped: {exc}")
                else:
                    stop, is_best = vanilla_stopper.step(monitor_value, vanilla_model)
                    if is_best:
                        vanilla_best_epoch = epoch
                        best_val_metrics_vanilla = val_metrics_vanilla_epoch
                    if stop:
                        vanilla_stopped_early = True
                        print(f"[VanillaTransformer] Early stopping at epoch {epoch} | best {vanilla_monitor_metric.upper()}={vanilla_stopper.best_score:.3f} @ epoch {vanilla_best_epoch}")
                        break
            else:
                try:
                    monitor_value = get_monitor_value(val_metrics_vanilla_epoch, vanilla_monitor_metric)
                except ValueError as exc:
                    print(f"[VanillaTransformer] Monitoring skipped: {exc}")
                else:
                    if not EarlyStopper._is_nan(monitor_value):
                        improved = False
                        if vanilla_best_score is None:
                            improved = True
                        elif vanilla_monitor_mode == 'max':
                            improved = monitor_value > (vanilla_best_score + args.vanilla_transformer_min_delta)
                        else:
                            improved = monitor_value < (vanilla_best_score - args.vanilla_transformer_min_delta)
                        if improved:
                            vanilla_best_score = monitor_value
                            vanilla_best_epoch = epoch
                            best_val_metrics_vanilla = val_metrics_vanilla_epoch
                            vanilla_best_state = copy.deepcopy(vanilla_model.state_dict())

        if vanilla_stopper:
            vanilla_stopper.restore_best(vanilla_model)
            if vanilla_best_epoch:
                status = "stopped early" if vanilla_stopped_early else "completed"
                print(f"[VanillaTransformer] {status}; best {vanilla_monitor_metric.upper()}={vanilla_stopper.best_score:.3f} @ epoch {vanilla_best_epoch}")
        else:
            if vanilla_best_state is not None:
                vanilla_model.load_state_dict(vanilla_best_state)

        if best_val_metrics_vanilla is None:
            best_val_metrics_vanilla = eval_transformer(vanilla_model, val_loader, device)

        final_metrics_vanilla = eval_transformer(vanilla_model, test_loader, device)
        if not best_test_metrics_vanilla_overall:
            best_test_metrics_vanilla_overall = {
                'auroc': final_metrics_vanilla['auroc'],
                'prauc': final_metrics_vanilla['prauc']
            }
        print("[VanillaTransformer] Best checkpoint evaluation")
        print_metrics("    Val", best_val_metrics_vanilla)
        print_metrics("    Test", final_metrics_vanilla)
        print(
            "    Test (best overall) "
            f"AUROC={fmt_metric_str(best_test_metrics_vanilla_overall.get('auroc', float('nan')))}, "
            f"PRAUC={fmt_metric_str(best_test_metrics_vanilla_overall.get('prauc', float('nan')))}"
        )

        row['auroc_vanilla_at_val'] = round_metric(final_metrics_vanilla['auroc'])
        row['auroc_vanilla_best'] = round_metric(best_test_metrics_vanilla_overall.get('auroc', float('nan')))
        row['prauc_vanilla_at_val'] = round_metric(final_metrics_vanilla['prauc'])
        row['prauc_vanilla_best'] = round_metric(best_test_metrics_vanilla_overall.get('prauc', float('nan')))

    # CDTrans
    model = CDTransModel(d_model=args.d_model,
                         n_heads=args.n_heads,
                         n_layers=args.n_layers,
                         num_classes=2,
                         f_dim=X_train.shape[1],
                         dropout=args.dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    monitor_metric = args.cdtrans_early_stop_metric.strip().lower()
    monitor_mode = 'min' if monitor_metric.endswith('loss') else 'max'
    pretrain_stopper = EarlyStopper(args.cdtrans_pretrain_patience,
                                    mode=monitor_mode,
                                    min_delta=args.cdtrans_early_stop_min_delta) if args.cdtrans_pretrain_patience >= 0 else None
    adapt_stopper = EarlyStopper(args.cdtrans_adapt_patience,
                                 mode=monitor_mode,
                                 min_delta=args.cdtrans_early_stop_min_delta) if args.cdtrans_adapt_patience >= 0 else None
    pretrain_best_epoch = 0
    adapt_best_epoch = 0
    pretrain_stopped_early = False
    adapt_stopped_early = False
    pretrain_val_metrics: Optional[Dict[str, float]] = None
    pretrain_test_metrics: Optional[Dict[str, float]] = None
    best_val_metrics_adapt: Optional[Dict[str, float]] = None
    best_test_metrics_adapt: Optional[Dict[str, float]] = None
    best_test_metrics_cd_overall: Dict[str, float] = {}

    print()
    print("[CDTrans] Pretraining...")
    for epoch in range(1, args.cdtrans_pretrain_epochs + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits, _ = model.train_source(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        loss_epoch = running_loss / len(train_loader.dataset)
        train_pre = eval_cdtrans(model, train_loader, device)
        val_pre = eval_cdtrans(model, val_loader, device)
        print(f"[CDTrans][Pretrain] Epoch {epoch}/{args.cdtrans_pretrain_epochs} Loss={loss_epoch:.3f}")
        print_metrics("    Train", train_pre)
        print_metrics("    Val", val_pre)

        if pretrain_stopper:
            try:
                monitor_value = get_monitor_value(val_pre, monitor_metric)
            except ValueError as exc:
                print(f"[CDTrans][Pretrain] Early stopping check skipped: {exc}")
            else:
                stop, is_best = pretrain_stopper.step(monitor_value, model)
                if is_best:
                    pretrain_best_epoch = epoch
                if stop:
                    pretrain_stopped_early = True
                    print(f"[CDTrans][Pretrain] Early stopping at epoch {epoch} | best {monitor_metric.upper()}={pretrain_stopper.best_score:.3f} @ epoch {pretrain_best_epoch}")
                    break

    if pretrain_stopper:
        pretrain_stopper.restore_best(model)
        if pretrain_best_epoch:
            status = "stopped early" if pretrain_stopped_early else "completed"
            print(f"[CDTrans][Pretrain] {status}; best {monitor_metric.upper()}={pretrain_stopper.best_score:.3f} @ epoch {pretrain_best_epoch}")

    pretrain_val_metrics = eval_cdtrans(model, val_loader, device)
    pretrain_test_metrics = eval_cdtrans(model, test_loader, device)
    print("[CDTrans][Pretrain] Best checkpoint evaluation")
    print_metrics("    Val", pretrain_val_metrics)
    print_metrics("    Test", pretrain_test_metrics)
    row['auroc_cdtrans_pretrain'] = round_metric(pretrain_test_metrics['auroc'])
    row['prauc_cdtrans_pretrain'] = round_metric(pretrain_test_metrics['prauc'])
    update_best_metrics(best_test_metrics_cd_overall,
                        {
                            'auroc': pretrain_test_metrics['auroc'],
                            'prauc': pretrain_test_metrics['prauc']
                        })

    print("[CDTrans] Adaptation...")
    for epoch in range(1, args.cdtrans_adapt_epochs + 1):
        pseudo = update_pseudo(model, test_loader, device)
        test_ds.pseudo = pseudo.to(torch.long)
        test_ds.use_pseudo = True
        matched_loader = create_pairs(model, train_ds, test_ds, args.batch_size, device, num_classes=2)

        model.train()
        running_loss = 0.0
        for (src_item, tgt_item) in matched_loader:
            (x_src, y_src, _), (x_tgt, y_pseudo, _) = src_item, tgt_item
            x_src = x_src.to(device)
            y_src = y_src.to(device)
            x_tgt = x_tgt.to(device)
            y_pseudo = y_pseudo.to(device)

            logits_src, logits_tgt, logits_cross, _, _, _ = model(x_src, x_tgt)
            loss_src = criterion(logits_src, y_src)
            loss_tgt = criterion(logits_tgt, y_pseudo)
            teacher = torch.clamp(F.softmax(logits_cross, dim=1), min=1e-6)
            loss_distill = F.kl_div(F.log_softmax(logits_tgt, dim=1), teacher, reduction='batchmean')
            loss = args.lambda_src * loss_src + args.lambda_tgt * loss_tgt + args.lambda_distill * loss_distill
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_src.size(0)

        loss_epoch = running_loss / len(matched_loader.dataset)
        val_metrics_cd = eval_cdtrans(model, val_loader, device)
        test_metrics_cd = eval_cdtrans(model, test_loader, device)
        print(f"[CDTrans][Adapt] Epoch {epoch}/{args.cdtrans_adapt_epochs} Loss={loss_epoch:.3f}")
        print_metrics("    Val", val_metrics_cd)
        print_metrics("    Test", test_metrics_cd)
        update_best_metrics(best_test_metrics_cd_overall,
                            {
                                'auroc': test_metrics_cd['auroc'],
                                'prauc': test_metrics_cd['prauc']
                            })

        if adapt_stopper:
            try:
                monitor_value = get_monitor_value(val_metrics_cd, monitor_metric)
            except ValueError as exc:
                print(f"[CDTrans][Adapt] Early stopping check skipped: {exc}")
            else:
                stop, is_best = adapt_stopper.step(monitor_value, model)
                if is_best:
                    adapt_best_epoch = epoch
                    best_val_metrics_adapt = val_metrics_cd
                    best_test_metrics_adapt = test_metrics_cd
                if stop:
                    adapt_stopped_early = True
                    print(f"[CDTrans][Adapt] Early stopping at epoch {epoch} | best {monitor_metric.upper()}={adapt_stopper.best_score:.3f} @ epoch {adapt_best_epoch}")
                    break
        else:
            if best_val_metrics_adapt is None:
                best_val_metrics_adapt = val_metrics_cd
                best_test_metrics_adapt = test_metrics_cd

    if adapt_stopper:
        adapt_stopper.restore_best(model)
        if adapt_best_epoch:
            status = "stopped early" if adapt_stopped_early else "completed"
            print(f"[CDTrans][Adapt] {status}; best {monitor_metric.upper()}={adapt_stopper.best_score:.3f} @ epoch {adapt_best_epoch}")
            if best_val_metrics_adapt is not None:
                print_metrics("    Best Val (snapshot)", best_val_metrics_adapt)
            if best_test_metrics_adapt is not None:
                print_metrics("    Best Test (snapshot)", best_test_metrics_adapt)

    test_ds.use_pseudo = False
    test_ds.pseudo = None
    final_metrics_cd = eval_cdtrans(model, test_loader, device)
    if best_val_metrics_adapt is None:
        best_val_metrics_adapt = eval_cdtrans(model, val_loader, device)
    if best_test_metrics_adapt is None:
        best_test_metrics_adapt = final_metrics_cd
    if not best_test_metrics_cd_overall:
        best_test_metrics_cd_overall = {
            'auroc': final_metrics_cd['auroc'],
            'prauc': final_metrics_cd['prauc']
        }

    row['auroc_cdtrans_at_val'] = round_metric(best_test_metrics_adapt['auroc'])
    row['auroc_cdtrans_best'] = round_metric(best_test_metrics_cd_overall.get('auroc', float('nan')))
    row['prauc_cdtrans_at_val'] = round_metric(best_test_metrics_adapt['prauc'])
    row['prauc_cdtrans_best'] = round_metric(best_test_metrics_cd_overall.get('prauc', float('nan')))

    print(
        "    [CDTrans] Test (best overall) "
        f"AUROC={fmt_metric_str(best_test_metrics_cd_overall.get('auroc', float('nan')))}, "
        f"PRAUC={fmt_metric_str(best_test_metrics_cd_overall.get('prauc', float('nan')))}"
    )

    summary_entries = [("LightGBM", test_metrics)]
    if final_metrics_vanilla is not None:
        summary_entries.append(("VanillaTransformer", final_metrics_vanilla))
    if pretrain_test_metrics is not None:
        summary_entries.append(("CDTrans-Pretrain", pretrain_test_metrics))
    summary_entries.append(("CDTrans-Adapted", final_metrics_cd))

    print()
    print("[Split Summary]")
    for label, metrics in summary_entries:
        print_metrics(f"[{label}]", metrics)

    return row


def run_experiment(args: argparse.Namespace) -> None:
    X, y, users = load_dataset(Path(args.dataset_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rows: List[Dict[str, Union[float, str]]] = []

    if args.mode == 'kfold' and args.output_csv == 'loso_cdtrans_vs_lgbm.csv':
        args.output_csv = 'kfold_cdtrans_vs_lgbm.csv'

    if args.mode == 'loso':
        unique_users = np.unique(users)
        for target_user in unique_users:
            test_mask = users == target_user
            X_test, y_test = X[test_mask], y[test_mask]
            X_source, y_source = X[~test_mask], y[~test_mask]

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
            train_idx, val_idx = next(splitter.split(X_source, y_source))
            X_train, y_train = X_source[train_idx], y_source[train_idx]
            X_val, y_val = X_source[val_idx], y_source[val_idx]

            row = evaluate_split(str(target_user), X_train, y_train, X_val, y_val, X_test, y_test, args, device)
            rows.append(row)
            persist_results(rows, args.output_csv)
    elif args.mode == 'kfold':
        skf = StratifiedKFold(n_splits=args.kfold_splits, shuffle=True, random_state=args.seed)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_full, y_train_full = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
            train_inner_idx, val_idx = next(splitter.split(X_train_full, y_train_full))
            X_train, y_train = X_train_full[train_inner_idx], y_train_full[train_inner_idx]
            X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

            row = evaluate_split(f"fold_{fold_idx}", X_train, y_train, X_val, y_val, X_test, y_test, args, device)
            rows.append(row)
            persist_results(rows, args.output_csv)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    df = build_results_dataframe(rows)
    print("=" * 80)
    print("Experiment complete. Summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("No numeric metrics recorded.")
    print("Results saved to", args.output_csv)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LGBM and CDTrans under LOSO or K-fold protocols")
    parser.add_argument('--mode', type=str, choices=['loso', 'kfold'], default='loso',
                        help='Evaluation protocol to run')
    parser.add_argument('--kfold-splits', type=int, default=5,
                        help='Number of folds for k-fold evaluation')
    parser.add_argument('--dataset-path', type=str, default='/home/iclab/minseo/Ubicomp/selected_users_dataset/reduced_49features_normalized.pkl')
    parser.add_argument('--output-csv', type=str, default='loso_cdtrans_vs_lgbm.csv')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lgbm-trials', type=int, default=30)
    parser.add_argument('--lgbm-num-threads', type=int, default=-1,
                        help='Number of threads for LightGBM (-1 uses all available cores)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--run-vanilla-transformer', action='store_true',
                        help='Enable training/evaluation of a vanilla transformer baseline')
    parser.add_argument('--vanilla-transformer-epochs', type=int, default=1000)
    parser.add_argument('--vanilla-transformer-patience', type=int, default=100,
                        help='Patience for vanilla transformer early stopping (set <0 to disable)')
    parser.add_argument('--vanilla-transformer-metric', type=str, default='auroc',
                        help='Validation metric to monitor for vanilla transformer (acc, auroc, prauc, f1)')
    parser.add_argument('--vanilla-transformer-min-delta', type=float, default=0.0,
                        help='Minimum improvement required for vanilla transformer early stopping')
    parser.add_argument('--cdtrans-pretrain-epochs', type=int, default=1000)
    parser.add_argument('--cdtrans-adapt-epochs', type=int, default=100)
    parser.add_argument('--cdtrans-pretrain-patience', type=int, default=50,
                        help='Patience for CDTrans pretraining early stopping (set <0 to disable)')
    parser.add_argument('--cdtrans-adapt-patience', type=int, default=50,
                        help='Patience for CDTrans adaptation early stopping (set <0 to disable)')
    parser.add_argument('--cdtrans-early-stop-metric', type=str, default='auroc',
                        help="Validation metric to monitor for CDTrans early stopping (acc, auroc, prauc, f1)")
    parser.add_argument('--cdtrans-early-stop-min-delta', type=float, default=0.0,
                        help='Minimum change required to qualify as an improvement for early stopping')
    parser.add_argument('--lambda-src', type=float, default=0.3)
    parser.add_argument('--lambda-tgt', type=float, default=1.0)
    parser.add_argument('--lambda-distill', type=float, default=1.0)
    args = parser.parse_args()

    run_experiment(args)


if __name__ == '__main__':
    main()
