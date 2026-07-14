"""ADDA: Adversarial Discriminative Domain Adaptation (Tzeng et al., 2017)."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models import train_torch_model

from .._da_helpers import (
    DAModel,
    EarlyStopTracker,
    _build_loaders,
    _evaluate_val,
    _finalize_training_metadata,
    _infinite_iterator,
)
from ..da_tllib_losses import DomainDiscriminator


class ADDA(nn.Module):
    """
    Adversarial Discriminative Domain Adaptation (Tzeng et al., 2017).
    Separate source/target encoders with adversarial discriminator.
    """

    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(ADDA, self).__init__()
        self.hparams = hparams if hparams else {}

        # Source model (encoder + classifier)
        self.source_model = DAModel(input_dim, num_classes, hparams)

        # Target encoder initialized from source encoder
        self.target_encoder = copy.deepcopy(self.source_model.feature_extractor)

        # Domain discriminator (binary, sigmoid output)
        feature_dim = self.source_model.feature_extractor.output_dim
        disc_hidden = self.hparams.get("disc_hidden", 1024)
        self.discriminator = DomainDiscriminator(
            feature_dim, disc_hidden, batch_norm=True, sigmoid=True
        )

    def predict(self, x):
        feat = self.target_encoder(x)
        return self.source_model.classifier(feat)

    def forward(self, x):
        return self.predict(x)


def train_adda(
    model,
    X_train,
    y_train,
    d_train,
    X_val,
    y_val,
    d_val,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    patience=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    X_target=None,
):
    """
    ADDA training:
      1) Pretrain source encoder+classifier on labeled source.
      2) Adversarially train target encoder to fool discriminator.
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("ADDA requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    model.discriminator = model.discriminator.to(device)

    # Phase 1: source pretraining
    model.source_model = train_torch_model(
        model.source_model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        device=device,
    )

    # Init target encoder from source
    model.target_encoder.load_state_dict(model.source_model.feature_extractor.state_dict())

    # Freeze source encoder and classifier
    for p in model.source_model.parameters():
        p.requires_grad = False

    weight_decay = model.hparams.get("weight_decay", 0.0)
    disc_lr = model.hparams.get("discriminator_lr", lr)
    opt_d = torch.optim.Adam(
        model.discriminator.parameters(), lr=disc_lr, weight_decay=weight_decay
    )
    opt_t = torch.optim.Adam(model.target_encoder.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()

    # Dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    target_dataset = torch.utils.data.TensorDataset(torch.tensor(X_target, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    target_iter = _infinite_iterator(target_loader)

    best_val_score = -float("inf")
    best_model_state = None
    patience_counter = 0
    source_pretrain_info = dict(getattr(model.source_model, "_training_info", {}))
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="ADDA Training")
    for epoch in epoch_iterator:
        model.target_encoder.train()
        model.discriminator.train()
        train_loss = 0.0

        for X_s, _ in train_loader:
            X_s = X_s.to(device, non_blocking=True)
            (X_t,) = next(target_iter)
            X_t = X_t.to(device, non_blocking=True)

            # 1) Train discriminator
            with torch.no_grad():
                f_s = model.source_model.feature_extractor(X_s)
            f_t = model.target_encoder(X_t)

            d_in = torch.cat([f_s, f_t], dim=0)
            d_out = model.discriminator(d_in)
            d_labels = torch.cat(
                [
                    torch.ones((f_s.size(0), 1), device=device),
                    torch.zeros((f_t.size(0), 1), device=device),
                ],
                dim=0,
            )

            opt_d.zero_grad()
            loss_d = bce(d_out, d_labels)
            loss_d.backward()
            opt_d.step()

            # 2) Train target encoder to fool discriminator
            f_t = model.target_encoder(X_t)
            d_out_t = model.discriminator(f_t)
            fool_labels = torch.ones((f_t.size(0), 1), device=device)
            opt_t.zero_grad()
            loss_t = bce(d_out_t, fool_labels)
            loss_t.backward()
            opt_t.step()

            train_loss += loss_d.item() + loss_t.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(
                    f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})"
                )
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {
            "Loss": f"{train_loss:.4f}",
            "Val Loss": f"{val_loss:.4f}",
            "Val AUC": f"{val_auroc:.4f}",
        }
        epoch_iterator.set_postfix(postfix)
        epoch_history.append(
            {
                "epoch": epoch_num,
                "train_loss": round(float(train_loss), 6),
                "val_loss": round(float(val_loss), 6),
                "val_auroc": round(float(val_auroc), 6),
            }
        )

    if best_model_state:
        model.load_state_dict(best_model_state)

    _finalize_training_metadata(
        model,
        optimizer=f"{opt_d.__class__.__name__}+{opt_t.__class__.__name__}",
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        extra={
            "discriminator_lr": disc_lr,
            "phases": {
                "source_pretrain": source_pretrain_info,
                "target_adaptation": {"epochs_ran": epochs_ran},
            },
        },
    )
    return model
