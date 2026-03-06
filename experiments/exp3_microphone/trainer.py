"""
Training loop for Experiment 3: fine-tune on recorded (NF), early stop on val (NF), evaluate on test (MF).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import LR, MAX_EPOCHS, PATIENCE, T_MAX, WEIGHT_DECAY, NUM_CLASSES

log = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    best_epoch: int
    best_val_f1: float
    total_time_s: float


def _macro_f1(preds: np.ndarray, targets: np.ndarray, n_classes: int = NUM_CLASSES) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(targets, preds, average="macro", zero_division=0))


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int = NUM_CLASSES,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    n = len(targets)
    return {
        "loss": total_loss / n,
        "acc": float((preds == targets).mean()),
        "f1": _macro_f1(preds, targets, n_classes),
        "preds": preds,
        "targets": targets,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    t_max: int = T_MAX,
) -> TrainHistory:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=t_max)

    best_val_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * y.size(0)
        n_train = len(train_loader.dataset)
        train_loss /= n_train
        scheduler.step()

        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        val_f1 = val_metrics["f1"]
        val_acc = val_metrics["acc"]

        log.info(
            "Ep %3d | trn_loss %.4f | val_acc %.3f val_f1 %.3f | lr %.2e",
            epoch, train_loss, val_acc, val_f1, optimiser.param_groups[0]["lr"],
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            log.info("Early stop at epoch %d (best %d, f1 %.4f)", epoch, best_epoch, best_val_f1)
            break

    total_time_s = time.perf_counter() - t0
    return TrainHistory(best_epoch=best_epoch, best_val_f1=best_val_f1, total_time_s=total_time_s)
