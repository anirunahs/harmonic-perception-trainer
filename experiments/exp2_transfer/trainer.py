"""
Training loop for Experiment 2 with differential learning rates,
automatic mixed precision (AMP), and early stopping.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import MIN_DELTA, T_MAX_TRANSFER, USE_AMP

log = logging.getLogger(__name__)


# History dataclasses

@dataclass
class EpochMetrics:
    train_loss: float = 0.0
    train_acc: float = 0.0
    train_dev_loss: float = 0.0
    train_dev_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    val_f1: float = 0.0
    lr_backbone: float = 0.0
    lr_head: float = 0.0


@dataclass
class TrainHistory:
    epochs: List[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0
    total_time_s: float = 0.0


# Helpers

def _macro_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(f1_score(targets, preds, average="macro", zero_division=0))


@torch.no_grad()
def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(x)
            total_loss += criterion(logits, y).item() * y.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(y.cpu().numpy())

    if was_training:
        model.train()

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    n = len(targets)
    return {
        "loss": total_loss / n,
        "acc": float((preds == targets).mean()),
        "f1": _macro_f1(preds, targets),
    }


# Trainer

class TransferTrainer:
    """Trains a TransferModel or a plain nn.Module to convergence."""

    def __init__(
        self,
        model: nn.Module,
        loaders: Dict[str, DataLoader],
        device: torch.device,
        backbone_lr: float = 0.0,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        t_max: int = T_MAX_TRANSFER,
    ):
        self.model = model.to(device)
        self.loaders = loaders
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience

        self.criterion = nn.CrossEntropyLoss()

        param_groups = self._build_param_groups(
            model, backbone_lr, head_lr)
        self.optimiser = AdamW(param_groups, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimiser, T_max=t_max)

        self.use_amp = USE_AMP and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    @staticmethod
    def _build_param_groups(model, backbone_lr, head_lr):
        if hasattr(model, "backbone") and hasattr(model, "head"):
            groups = [{"params": list(model.head.parameters()),
                       "lr": head_lr}]
            bb = [p for p in model.backbone.parameters() if p.requires_grad]
            if bb and backbone_lr > 0:
                groups.append({"params": bb, "lr": backbone_lr})
            return groups
        return [{"params": list(model.parameters()), "lr": head_lr}]

    def _train_one_epoch(self) -> float:
        self.model.train()
        running, n = 0.0, 0
        for x, y in self.loaders["train"]:
            x, y = x.to(self.device), y.to(self.device)
            self.optimiser.zero_grad()
            with torch.autocast("cuda",
                                enabled=self.use_amp):
                loss = self.criterion(self.model(x), y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimiser)
            self.scaler.update()
            running += loss.item() * y.size(0)
            n += y.size(0)
        return running / n

    @staticmethod
    def _state_to_cpu(state_dict: dict) -> dict:
        return {k: v.cpu() for k, v in state_dict.items()}

    def fit(self) -> TrainHistory:
        history = TrainHistory()
        best_state = None
        wait = 0
        t0 = time.time()

        for epoch in range(1, self.max_epochs + 1):
            trn_loss = self._train_one_epoch()

            m = EpochMetrics()
            m.lr_head = self.optimiser.param_groups[0]["lr"]
            if len(self.optimiser.param_groups) > 1:
                m.lr_backbone = self.optimiser.param_groups[1]["lr"]

            m.train_loss = trn_loss

            if "train_dev" in self.loaders:
                td = _evaluate_loader(
                    self.model, self.loaders["train_dev"],
                    self.criterion, self.device, self.use_amp)
                m.train_dev_loss, m.train_dev_acc = td["loss"], td["acc"]

            v = _evaluate_loader(self.model, self.loaders["val"],
                                 self.criterion, self.device, self.use_amp)
            m.val_loss, m.val_acc, m.val_f1 = v["loss"], v["acc"], v["f1"]

            history.epochs.append(m)
            self.scheduler.step()

            if m.val_f1 > history.best_val_f1 + MIN_DELTA:
                history.best_val_f1 = m.val_f1
                history.best_epoch = epoch
                best_state = self._state_to_cpu(self.model.state_dict())
                wait = 0
            else:
                wait += 1

            if epoch % 3 == 0 or epoch == 1:
                log.info(
                    "Ep %3d | trn_loss %.3f | td %.3f | val_f1 %.3f "
                    "val_acc %.3f | lr_h %.1e lr_b %.1e",
                    epoch, m.train_loss, m.train_dev_acc, m.val_f1,
                    m.val_acc, m.lr_head, m.lr_backbone,
                )

            if wait >= self.patience:
                log.info("Early stop at epoch %d (best %d, f1 %.4f)",
                         epoch, history.best_epoch, history.best_val_f1)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        del best_state

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        history.total_time_s = time.time() - t0
        return history
