"""Transformer-based training pipeline for domain adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .cdtrans import (
    DEFAULT_PRIORITY_PREFIXES,
    FeatureBlockTokenizer,
    FeedForwardBlock,
    LayerNormalization,
    MultiHeadAttention,
    ResidualConnection,
)
from .common import ArrayDataset, safe_accuracy, safe_auc, safe_auprc


@dataclass
class TransformerConfig:
    input_dim: int
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    feature_names: Optional[Tuple[str, ...]] = None
    max_block_size: int = 128
    block_grouping: str = "prefix_slot"
    attention_type: str = "linear"
    linear_attn_features: int = 64
    priority_prefixes: Tuple[str, ...] = DEFAULT_PRIORITY_PREFIXES
    pretrain_epochs: int = 300
    # pretrain_epochs: int = 15
    finetune_epochs: int = 50
    adapt_epochs: int = 5
    batch_size: int = 256
    pretrain_lr: float = 1e-3
    finetune_lr: float = 5e-4
    adapt_lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: Optional[str] = None
    use_cosine_scheduler: bool = False
    cosine_min_lr: float = 1e-5
    use_plateau_scheduler: bool = False
    plateau_factor: float = 0.5
    plateau_patience: int = 10
    warmup_epochs: int = 0


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


def _fmt_metric(value: float) -> str:
    if value is None or np.isnan(value):
        return "nan"
    return f"{value:.3f}"


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        attention_type: str,
        linear_attn_features: int,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attention_type=attention_type,
            nb_features=linear_attn_features,
        )
        self.residual1 = ResidualConnection(dropout)
        self.residual2 = ResidualConnection(dropout)
        self.ff = FeedForwardBlock(d_model, d_ff=4 * d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual1(x, lambda t: self.mha(t, t, t))
        x = self.residual2(x, self.ff)
        return x


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim: int, config: TransformerConfig):
        super().__init__()
        self.input_dim = input_dim
        self.tokenizer = FeatureBlockTokenizer(
            input_dim,
            config.d_model,
            feature_names=config.feature_names,
            max_block_size=config.max_block_size,
            grouping=config.block_grouping,
            priority_prefixes=config.priority_prefixes,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.dropout,
                config.attention_type,
                config.linear_attn_features,
            )
            for _ in range(config.num_layers)
        ])
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        tokens = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        h = torch.cat([cls_tokens, tokens], dim=1)
        for layer in self.layers:
            h = layer(h)
        cls_output = self.norm(h[:, 0, :])
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

    def _evaluate_metrics(self, model: FeatureTransformer, dataset: Optional[ArrayDataset]) -> Dict[str, float]:
        if dataset is None or dataset.X.size == 0:
            return {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
        preds = self._predict(model, dataset)
        return {
            "auroc": safe_auc(dataset.y, preds),
            "accuracy": safe_accuracy(dataset.y, preds),
            "auprc": safe_auprc(dataset.y, preds),
        }

    def _train_stage(
        self,
        model: FeatureTransformer,
        dataset: ArrayDataset,
        epochs: int,
        lr: float,
        stage_name: str,
        *,
        val_dataset: Optional[ArrayDataset] = None,
    ) -> Tuple[float, int]:
        if epochs <= 0 or dataset.X.size == 0:
            return 0.0, 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.config.weight_decay)
        scheduler = None
        cosine_epochs = max(1, epochs - max(0, self.config.warmup_epochs))
        if self.config.use_cosine_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=self.config.cosine_min_lr,
            )
        elif self.config.use_plateau_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.plateau_factor,
                patience=max(1, self.config.plateau_patience),
                min_lr=self.config.cosine_min_lr,
            )
        loader = self._make_loader(dataset, shuffle=True)
        model.train()
        start = perf_counter()
        epoch_counter = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_loss = float("inf")
        # patience = max(1, epochs // 4) if val_dataset is not None and val_dataset.X.size > 0 else None
        patience = None
        epochs_without_improvement = 0
        iterator = tqdm(
            range(1, epochs + 1),
            desc=f"[Transformer][{stage_name}]",
            leave=False,
            dynamic_ncols=True,
            disable=False,
        )
        for epoch in iterator:
            if self.config.warmup_epochs and epoch <= self.config.warmup_epochs:
                warmup_scale = max(1e-3, epoch / float(self.config.warmup_epochs))
                effective_lr = lr * warmup_scale
                for param_group in optimizer.param_groups:
                    param_group["lr"] = effective_lr
            elif scheduler is None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            epoch_loss = 0.0
            sample_count = 0
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
                epoch_loss += loss.item() * xb.size(0)
                sample_count += xb.size(0)
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
                val_metrics = self._evaluate_metrics(model, val_dataset)
            else:
                val_metrics = {"auroc": float("nan"), "accuracy": float("nan"), "auprc": float("nan")}
            train_metrics = self._evaluate_metrics(model, dataset)
            avg_loss = epoch_loss / sample_count if sample_count else float("nan")
            iterator.set_postfix({
                "epoch": epoch,
                "loss": f"{avg_loss:.4f}" if not np.isnan(avg_loss) else "nan",
                "train_auc": _fmt_metric(train_metrics["auroc"]),
                "val_auc": _fmt_metric(val_metrics["auroc"]),
            })
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor = val_loss if val_dataset is not None and val_dataset.X.size > 0 else avg_loss
                    scheduler.step(monitor if monitor is not None else avg_loss)
                else:
                    if not (self.config.warmup_epochs and epoch < self.config.warmup_epochs):
                        scheduler.step()
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
                stage_name="Pretrain",
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
            stage_name="Finetune",
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
                stage_name="Adapt",
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
