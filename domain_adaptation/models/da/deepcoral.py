"""DeepCORAL: Correlation Alignment for Deep Domain Adaptation."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .._da_helpers import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DAModel,
    EarlyStopTracker,
    _build_loaders,
    _evaluate_val,
    _infinite_iterator,
)
from ..da_tllib_losses import CorrelationAlignmentLoss


class DeepCORAL(DAModel):
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(DeepCORAL, self).__init__(input_dim, num_classes, hparams)

    # Standard forward uses DAModel.predict


def train_deepcoral(
    model,
    X_train,
    y_train,
    d_train,
    X_val,
    y_val,
    d_val,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    device: str = DEFAULT_DEVICE,
    X_target=None,
):
    """
    DeepCORAL (TLL-style): L = L_cls(source) + lambda * CORAL(f_s, f_t).
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError(
            "DeepCORAL requires X_target for UDA. Use --uda to provide target samples."
        )

    model = model.to(device)
    weight_decay = model.hparams.get("weight_decay", 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_criterion = nn.CrossEntropyLoss()
    coral = CorrelationAlignmentLoss()
    trade_off = model.hparams.get("coral_lambda", model.hparams.get("mmd_gamma", 1.0))

    # Dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )
    target_dataset = torch.utils.data.TensorDataset(torch.tensor(X_target, dtype=torch.float32))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    def infinite_iterator(loader):
        while True:
            for batch in loader:
                yield batch

    target_iter = infinite_iterator(target_loader)

    tracker = EarlyStopTracker(
        patience=patience,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )

    epoch_iterator = tqdm(range(epochs), desc="DeepCORAL Training")
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            (X_t,) = next(target_iter)
            X_t = X_t.to(device)

            optimizer.zero_grad()

            f_s = model.feature_extractor(X_s)
            f_t = model.feature_extractor(X_t)
            logits_s = model.classifier(f_s)

            cls_loss = class_criterion(logits_s, y_s)
            transfer_loss = coral(f_s, f_t)
            loss = cls_loss + trade_off * transfer_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model.predict(X_batch)
                val_loss += class_criterion(logits, y_batch).item()
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        try:
            val_auroc = roc_auc_score(val_targets, val_probs)
        except ValueError:
            val_auroc = 0.5

        if tracker.record(
            model,
            epoch_num=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_auroc=val_auroc,
            iterator=epoch_iterator,
        ):
            break

    return tracker.finalize(model, optimizer=optimizer)


# --- CGDM: Cross-Domain Gradient Discrepancy Minimization (CVPR 2021) ---
# Ported from /home/iclab/minseo/DomainAdaptation/CGDM with minimal changes.
