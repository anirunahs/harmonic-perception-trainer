"""
Training loop with early stopping, LR scheduling, and optional Optuna tuning.

Tracks *four* error metrics per epoch (train, train-dev, val, test-peek)
to support bias / variance / data-mismatch framework.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import (
    LR, MAX_EPOCHS, MIN_DELTA, PATIENCE, T_MAX, WEIGHT_DECAY,
    OPTUNA_BATCH_SIZES, OPTUNA_DROPOUT_RANGE,
    OPTUNA_LR_RANGE, OPTUNA_TRIALS, OPTUNA_WD_RANGE,
)
from .models import build_model

log = logging.getLogger(__name__)


# History dataclass

@dataclass
class EpochMetrics:
    train_loss: float = 0.0
    train_acc: float = 0.0
    train_dev_loss: float = 0.0
    train_dev_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    val_f1: float = 0.0
    lr: float = 0.0


@dataclass
class TrainHistory:
    epochs: List[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0
    total_time_s: float = 0.0


# Helpers

def _macro_f1(preds: np.ndarray, targets: np.ndarray, n_classes: int) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(targets, preds, average="macro", zero_division=0))


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int = 12,
) -> Dict[str, float]:
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
    }


# Trainer

class Trainer:
    """Trains one model to convergence and records full history."""

    def __init__(
        self,
        model: nn.Module,
        loaders: Dict[str, DataLoader],
        device: torch.device,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
        max_epochs: int = MAX_EPOCHS,
        patience: int = PATIENCE,
        t_max: int = T_MAX,
    ):
        self.model = model.to(device)
        self.loaders = loaders
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience

        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimiser, T_max=t_max)

    def _train_one_epoch(self) -> float:
        self.model.train()
        running = 0.0
        n = 0
        for x, y in self.loaders["train"]:
            x, y = x.to(self.device), y.to(self.device)
            self.optimiser.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            self.optimiser.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        return running / n

    def fit(self) -> TrainHistory:
        history = TrainHistory()
        best_state = None
        wait = 0
        t0 = time.time()

        for epoch in range(1, self.max_epochs + 1):
            self._train_one_epoch()

            m = EpochMetrics(lr=self.optimiser.param_groups[0]["lr"])

            tr = evaluate_loader(self.model, self.loaders["train"],
                                 self.criterion, self.device)
            m.train_loss, m.train_acc = tr["loss"], tr["acc"]

            if "train_dev" in self.loaders:
                td = evaluate_loader(self.model, self.loaders["train_dev"],
                                     self.criterion, self.device)
                m.train_dev_loss, m.train_dev_acc = td["loss"], td["acc"]

            v = evaluate_loader(self.model, self.loaders["val"],
                                self.criterion, self.device)
            m.val_loss, m.val_acc, m.val_f1 = v["loss"], v["acc"], v["f1"]

            history.epochs.append(m)
            self.scheduler.step()

            if m.val_f1 > history.best_val_f1 + MIN_DELTA:
                history.best_val_f1 = m.val_f1
                history.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    "Epoch %3d | trn_loss %.4f  trn_acc %.3f | "
                    "td_acc %.3f | val_f1 %.3f  val_acc %.3f | lr %.2e",
                    epoch, m.train_loss, m.train_acc,
                    m.train_dev_acc, m.val_f1, m.val_acc, m.lr,
                )

            if wait >= self.patience:
                log.info("Early stop at epoch %d (best %d, val_f1 %.4f)",
                         epoch, history.best_epoch, history.best_val_f1)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        history.total_time_s = time.time() - t0
        return history


# Optuna integration

def run_optuna(
    repr_name: str,
    splits,
    device: torch.device,
    n_trials: int = OPTUNA_TRIALS,
) -> Dict:
    """Run Optuna HPO and return the best trial's parameters."""
    try:
        import optuna
    except ImportError:
        log.warning("optuna not installed — skipping HPO, using defaults.")
        return {
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "dropout": 0.3, "batch_size": 32,
        }

    from .dataset import create_dataloaders

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", *OPTUNA_LR_RANGE, log=True)
        wd = trial.suggest_float("weight_decay", *OPTUNA_WD_RANGE, log=True)
        dr = trial.suggest_float("dropout", *OPTUNA_DROPOUT_RANGE)
        bs = trial.suggest_categorical("batch_size", list(OPTUNA_BATCH_SIZES))

        loaders = create_dataloaders(splits, batch_size=bs,
                                     repr_name=repr_name)
        model = build_model(repr_name, dropout=dr)

        trainer = Trainer(
            model, loaders, device,
            lr=lr, weight_decay=wd,
            max_epochs=80,
            patience=12,
        )
        history = trainer.fit()
        return history.best_val_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    log.info("Optuna best: %s  (val_f1 = %.4f)", best, study.best_value)
    return best
