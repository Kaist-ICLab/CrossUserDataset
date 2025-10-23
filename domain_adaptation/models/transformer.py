"""Transformer-based training pipeline for domain adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import ArrayDataset, safe_accuracy, safe_auc, safe_auprc


@dataclass
class TransformerConfig:
    input_dim: int
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    pretrain_epochs: int = 15
    finetune_epochs: int = 10
    adapt_epochs: int = 5
    batch_size: int = 256
    pretrain_lr: float = 1e-3
    finetune_lr: float = 5e-4
    adapt_lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: Optional[str] = None


@dataclass
class TransformerRunResult:
    train_auroc: float
    val_auroc: float
    test_auroc: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_auprc: float
    val_auprc: float
    test_auprc: float
    pretrain_val_auroc: float
    pretrain_val_accuracy: float
    pretrain_val_auprc: float
    stage_durations: Dict[str, float]
    stage_epochs: Dict[str, int]
    state_dict: dict


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim: int, config: TransformerConfig):
        super().__init__()
        self.input_dim = input_dim
        self.value_proj = nn.Linear(1, config.d_model)
        self.position_embedding = nn.Embedding(input_dim, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        seq = x.unsqueeze(-1)  # (batch, features, 1)
        token_embeddings = self.value_proj(seq)
        positions = torch.arange(self.input_dim, device=x.device)
        token_embeddings = token_embeddings + self.position_embedding(positions)[None, :, :]
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls_tokens, token_embeddings], dim=1)
        encoded = self.encoder(tokens)
        cls_output = self.norm(encoded[:, 0, :])
        logits = self.head(self.dropout(cls_output)).squeeze(-1)
        return logits


class TransformerPipeline:
    def __init__(self, config: TransformerConfig):
        self.config = config
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()

    def _build_model(self) -> FeatureTransformer:
        model = FeatureTransformer(self.config.input_dim, self.config)
        return model.to(self.device)

    def _make_loader(self, dataset: ArrayDataset, shuffle: bool) -> DataLoader:
        tensors = (
            torch.from_numpy(dataset.X.astype(np.float32)),
            torch.from_numpy(dataset.y.astype(np.float32)),
        )
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=self.config.batch_size, shuffle=shuffle, drop_last=False)

    def _evaluate_loss(self, model: FeatureTransformer, dataset: ArrayDataset) -> float:
        if dataset.X.size == 0:
            return float("inf")
        loader = self._make_loader(dataset, shuffle=False)
        model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = model(xb)
                loss = self.criterion(logits, yb)
                batch = xb.size(0)
                total_loss += loss.item() * batch
                total_count += batch
        model.train()
        if total_count == 0:
            return float("inf")
        return total_loss / total_count

    def _train_stage(
        self,
        model: FeatureTransformer,
        dataset: ArrayDataset,
        epochs: int,
        lr: float,
        *,
        val_dataset: Optional[ArrayDataset] = None,
    ) -> Tuple[float, int]:
        if epochs <= 0 or dataset.X.size == 0:
            return 0.0, 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.config.weight_decay)
        loader = self._make_loader(dataset, shuffle=True)
        model.train()
        start = perf_counter()
        epoch_counter = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_loss = float("inf")
        patience = max(1, epochs // 4) if val_dataset is not None and val_dataset.X.size > 0 else None
        epochs_without_improvement = 0
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = model(xb)
                loss = self.criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()
            epoch_counter += 1
            if val_dataset is not None and val_dataset.X.size > 0:
                val_loss = self._evaluate_loss(model, val_dataset)
                if val_loss < best_loss - 1e-5:
                    best_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if patience is not None and epochs_without_improvement >= patience:
                        break
        elapsed = perf_counter() - start
        if best_state is not None:
            model.load_state_dict(best_state)
        return elapsed, epoch_counter

    def _predict(self, model: FeatureTransformer, dataset: ArrayDataset) -> np.ndarray:
        if dataset.X.size == 0:
            return np.array([])
        loader = self._make_loader(dataset, shuffle=False)
        model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                preds.append(probs.cpu().numpy())
        return np.concatenate(preds)

    def run(
        self,
        *,
        seed: int,
        pretrain: Optional[ArrayDataset],
        pretrain_val: Optional[ArrayDataset],
        train: ArrayDataset,
        val: ArrayDataset,
        adapt: Optional[ArrayDataset],
        evaluation: ArrayDataset,
    ) -> TransformerRunResult:
        _seed_everything(seed)
        model = self._build_model()
        stage_durations: Dict[str, float] = {}
        stage_epochs: Dict[str, int] = {}
        pretrain_val_auroc = float("nan")
        pretrain_val_accuracy = float("nan")
        pretrain_val_auprc = float("nan")

        if pretrain is not None and pretrain.X.size > 0:
            duration, epochs_run = self._train_stage(
                model,
                pretrain,
                self.config.pretrain_epochs,
                self.config.pretrain_lr,
                val_dataset=pretrain_val,
            )
            stage_durations["pretrain_seconds"] = duration
            stage_epochs["pretrain_epochs"] = epochs_run
            if pretrain_val is not None and pretrain_val.X.size > 0:
                pretrain_val_pred = self._predict(model, pretrain_val)
                pretrain_val_auroc = safe_auc(pretrain_val.y, pretrain_val_pred)
                pretrain_val_accuracy = safe_accuracy(pretrain_val.y, pretrain_val_pred)
                pretrain_val_auprc = safe_auprc(pretrain_val.y, pretrain_val_pred)

        duration, epochs_run = self._train_stage(
            model,
            train,
            self.config.finetune_epochs,
            self.config.finetune_lr,
            val_dataset=val,
        )
        stage_durations["finetune_seconds"] = duration
        stage_epochs["finetune_epochs"] = epochs_run

        if adapt is not None and adapt.X.size > 0:
            duration, epochs_run = self._train_stage(
                model,
                adapt,
                self.config.adapt_epochs,
                self.config.adapt_lr,
                val_dataset=val,
            )
            stage_durations["adapt_seconds"] = duration
            stage_epochs["adapt_epochs"] = epochs_run

        train_pred = self._predict(model, train)
        val_pred = self._predict(model, val)
        test_pred = self._predict(model, evaluation)

        return TransformerRunResult(
            train_auroc=safe_auc(train.y, train_pred),
            val_auroc=safe_auc(val.y, val_pred),
            test_auroc=safe_auc(evaluation.y, test_pred),
            train_accuracy=safe_accuracy(train.y, train_pred),
            val_accuracy=safe_accuracy(val.y, val_pred),
            test_accuracy=safe_accuracy(evaluation.y, test_pred),
            train_auprc=safe_auprc(train.y, train_pred),
            val_auprc=safe_auprc(val.y, val_pred),
            test_auprc=safe_auprc(evaluation.y, test_pred),
            pretrain_val_auroc=pretrain_val_auroc,
            pretrain_val_accuracy=pretrain_val_accuracy,
            pretrain_val_auprc=pretrain_val_auprc,
            stage_durations=stage_durations,
            stage_epochs=stage_epochs,
            state_dict=model.state_dict(),
        )
