"""
CNN architectures for Experiment 1.

Two families share the same classifier head so the only variable is the
input representation:

    SpectrogramCNN  — 2-D conv blocks for Mel / CQT / HCQT
    WaveformCNN     — 1-D conv blocks for raw audio
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import (
    NUM_CLASSES,
    CNN2D_CHANNELS, CNN2D_KERNEL, CLASSIFIER_DIM,
    CNN1D_CHANNELS, CNN1D_KERNELS, CNN1D_FIRST_STRIDE,
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


class SpectrogramCNN(nn.Module):
    """4-block 2-D CNN — GAP — Dense classifier."""

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
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

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128-d embedding before the last linear layer."""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        for layer in list(self.classifier.children())[:-1]:
            x = layer(x)
        return x


class _ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 stride: int = 1, pool: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveformCNN(nn.Module):
    """1-D CNN on raw waveform (learned filterbank paradigm)."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        ch = CNN1D_CHANNELS
        ks = CNN1D_KERNELS
        self.features = nn.Sequential(
            _ConvBlock1D(1, ch[0], ks[0], stride=CNN1D_FIRST_STRIDE, pool=4),
            _ConvBlock1D(ch[0], ch[1], ks[1], pool=4),
            _ConvBlock1D(ch[1], ch[2], ks[2], pool=4),
            _ConvBlock1D(ch[2], ch[3], ks[3], pool=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1], CLASSIFIER_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(CLASSIFIER_DIM, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(repr_name: str, dropout: float = 0.3) -> nn.Module:
    if repr_name == "raw":
        return WaveformCNN(dropout=dropout)
    in_ch = 5 if repr_name == "hcqt" else 1
    return SpectrogramCNN(in_channels=in_ch, dropout=dropout)
