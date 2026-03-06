"""
HCQT CNN for Experiment 3.

Optionally load pretrained weights from Exp 1 checkpoint before fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .config import (
    NUM_CLASSES,
    CNN2D_CHANNELS,
    CNN2D_KERNEL,
    CLASSIFIER_DIM,
    HCQT_IN_CHANNELS,
    DROPOUT,
)


class _ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HCQT_CNN(nn.Module):
    """4-block 2-D CNN for HCQT (5 channels)."""

    def __init__(self, in_channels: int = HCQT_IN_CHANNELS, dropout: float = DROPOUT):
        super().__init__()
        channels = (in_channels,) + CNN2D_CHANNELS
        self.features = nn.Sequential(
            *[_ConvBlock2D(channels[i], channels[i + 1], CNN2D_KERNEL)
              for i in range(len(CNN2D_CHANNELS))]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CNN2D_CHANNELS[-1], CLASSIFIER_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(CLASSIFIER_DIM, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_hcqt_model(dropout: float = DROPOUT) -> HCQT_CNN:
    return HCQT_CNN(in_channels=HCQT_IN_CHANNELS, dropout=dropout)


def build_mel_model(dropout: float = DROPOUT) -> HCQT_CNN:
    """Same architecture as HCQT CNN but 1 channel (Mel)."""
    return HCQT_CNN(in_channels=1, dropout=dropout)


def build_cqt_model(dropout: float = DROPOUT) -> HCQT_CNN:
    """Same architecture as HCQT CNN but 1 channel (CQT)."""
    return HCQT_CNN(in_channels=1, dropout=dropout)


def build_model(representation: str, dropout: float = DROPOUT) -> HCQT_CNN:
    """Build model for hcqt, mel, or cqt."""
    if representation == "hcqt":
        return build_hcqt_model(dropout=dropout)
    if representation == "mel":
        return build_mel_model(dropout=dropout)
    if representation == "cqt":
        return build_cqt_model(dropout=dropout)
    raise ValueError("representation must be hcqt, mel, or cqt")


def load_pretrained(model: HCQT_CNN, checkpoint_path: Path, strict: bool = True) -> HCQT_CNN:
    """Load Exp 1 HCQT checkpoint into model."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
 
    new_state = {}
    for k, v in state.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[key] = v
    model.load_state_dict(new_state, strict=strict)
    return model
