"""
Models for Experiment 4: HCQT CNN (with embedding access for t-SNE/UMAP and Grad-CAM)
and FFT-based MLP baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from .config import (
    NUM_CLASSES,
    CNN2D_CHANNELS,
    CNN2D_KERNEL,
    CLASSIFIER_DIM,
    HCQT_IN_CHANNELS,
    DROPOUT,
    FFT_FEATURE_DIM,
    MLP_HIDDEN,
    MLP_DROPOUT,
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
    """4-block 2-D CNN for HCQT."""
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

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Output of penultimate layer for t-SNE/UMAP."""
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        return x


class FFT_MLP(nn.Module):
    """MLP on FFT numerical features."""

    def __init__(
        self,
        input_dim: int = FFT_FEATURE_DIM,
        hidden: Tuple[int, ...] = MLP_HIDDEN,
        num_classes: int = NUM_CLASSES,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()
        dims: List[int] = [input_dim] + list(hidden) + [num_classes]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Output of penultimate layer for t-SNE/UMAP."""
        for i in range(len(self.net) - 1):
            x = self.net[i](x)
        return x


def build_hcqt_cnn(dropout: float = DROPOUT) -> HCQT_CNN:
    return HCQT_CNN(in_channels=HCQT_IN_CHANNELS, dropout=dropout)


def load_hcqt_checkpoint(model: HCQT_CNN, checkpoint_path: Path, strict: bool = True) -> HCQT_CNN:
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


def build_fft_mlp(
    input_dim: int = FFT_FEATURE_DIM,
    hidden: Tuple[int, ...] = MLP_HIDDEN,
    dropout: float = MLP_DROPOUT,
) -> FFT_MLP:
    return FFT_MLP(input_dim=input_dim, hidden=hidden, num_classes=NUM_CLASSES, dropout=dropout)
