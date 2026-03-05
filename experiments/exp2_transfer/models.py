"""
Model architectures for Experiment 2.

Three model families:
  1. PANNs CNN14 backbone + interval classifier head
  2. AST (Audio Spectrogram Transformer) backbone + interval classifier head
  3. From-scratch CNN baseline (imported from Experiment 1)

Each pretrained backbone supports three fine-tuning strategies:
  - frozen:  all backbone parameters fixed; only classifier head trained
  - partial: last N blocks / layers unfrozen + classifier head
  - full:    entire model trainable with differential learning rates
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    AST_EMBED_DIM,
    AST_MODEL_NAME,
    AST_UNFREEZE_LAYERS,
    CHECKPOINT_DIR,
    CLASSIFIER_HIDDEN,
    DROPOUT,
    NUM_CLASSES,
    PANNS_CHECKPOINT_FNAME,
    PANNS_CHECKPOINT_URL,
    PANNS_EMBED_DIM,
    PANNS_UNFREEZE_BLOCKS,
)

log = logging.getLogger(__name__)


#  PANNs CNN14 — Architecture

class _ConvBlock(nn.Module):
    """Two-layer conv block matching the official PANNs CNN14 layout."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3),
                               padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor,
                pool_size: Tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class CNN14Backbone(nn.Module):
    """PANNs CNN14 backbone without the mel-spectrogram front-end and
    without the AudioSet classification head.

    Input : pre-computed log10-mel spectrogram, shape ``(B, 1, T, 64)``.
    Output: 2048-d embedding vector.
    """

    BLOCK_NAMES = [
        "conv_block1", "conv_block2", "conv_block3",
        "conv_block4", "conv_block5", "conv_block6",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048)
        self.embed_dim = PANNS_EMBED_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F=64)
        x = x.transpose(1, 3)                           # (B, 64, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)                           # (B, 1, T, 64)

        x = F.dropout(self.conv_block1(x, (2, 2)), 0.2, self.training)
        x = F.dropout(self.conv_block2(x, (2, 2)), 0.2, self.training)
        x = F.dropout(self.conv_block3(x, (2, 2)), 0.2, self.training)
        x = F.dropout(self.conv_block4(x, (2, 2)), 0.2, self.training)
        x = F.dropout(self.conv_block5(x, (2, 2)), 0.2, self.training)
        x = F.dropout(self.conv_block6(x, (1, 1)), 0.2, self.training)

        x = torch.mean(x, dim=3)                        # mean over mel
        x1, _ = torch.max(x, dim=2)                     # max over time
        x2 = torch.mean(x, dim=2)                       # mean over time
        x = x1 + x2                                     # (B, 2048)

        x = F.dropout(x, 0.5, self.training)
        x = F.relu_(self.fc1(x))
        return x                                         # (B, 2048)

    def get_blocks(self):
        return [getattr(self, n) for n in self.BLOCK_NAMES]


# Checkpoint download and weight loading

def _download_panns_checkpoint(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / PANNS_CHECKPOINT_FNAME
    if target_path.exists():
        log.info("PANNs checkpoint cached: %s", target_path)
        return target_path

    log.info("Downloading PANNs CNN14 checkpoint...")
    import urllib.request
    urllib.request.urlretrieve(PANNS_CHECKPOINT_URL, str(target_path))
    log.info("Saved → %s", target_path)
    return target_path


def load_panns_backbone() -> CNN14Backbone:
    """Build ``CNN14Backbone`` and load AudioSet-pretrained weights."""
    ckpt_path = _download_panns_checkpoint(CHECKPOINT_DIR)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    backbone = CNN14Backbone()

    filtered = {
        k: v for k, v in state_dict.items()
        if not k.startswith(("spectrogram_extractor", "logmel_extractor",
                             "fc_audioset"))
    }
    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    if missing:
        log.warning("Missing keys when loading PANNs: %s", missing)

    n_params = sum(p.numel() for p in backbone.parameters())
    log.info("PANNs CNN14 backbone loaded — %.1f M params.", n_params / 1e6)
    return backbone


#  AST (Audio Spectrogram Transformer)

class ASTBackbone(nn.Module):
    """Wrapper around HuggingFace ASTModel"""

    _ORIG_TIME_FRAMES = 1024
    _PATCH_SIZE = 16
    _TIME_STRIDE = 10
    _FREQ_STRIDE = 10
    _NUM_SPECIAL = 2  # CLS + distillation

    def __init__(self, model_name: str = AST_MODEL_NAME,
                 target_length: int | None = None) -> None:
        super().__init__()
        try:
            from transformers import ASTModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "AST requires the 'transformers' package. "
                "Install with: pip install transformers"
            ) from e
        from .config import AST_TARGET_LENGTH

        self.ast = ASTModel.from_pretrained(model_name)
        self.embed_dim = self.ast.config.hidden_size     # 768

        tgt = target_length or AST_TARGET_LENGTH
        self._resize_pos_embeddings(tgt)

    def _resize_pos_embeddings(self, target_length: int) -> None:
        pos = self.ast.embeddings.position_embeddings.data  # (1, N, D)
        D = pos.shape[-1]

        orig_pt = (self._ORIG_TIME_FRAMES - self._PATCH_SIZE) \
                  // self._TIME_STRIDE + 1                   # 101
        orig_pf = (128 - self._PATCH_SIZE) \
                  // self._FREQ_STRIDE + 1                   # 12

        new_pt = (target_length - self._PATCH_SIZE) \
                 // self._TIME_STRIDE + 1
        new_pf = orig_pf

        if orig_pt * orig_pf + self._NUM_SPECIAL == pos.shape[1] \
                and new_pt == orig_pt:
            return

        cls_dist = pos[:, :self._NUM_SPECIAL, :]             # (1, 2, D)
        patch_pos = pos[:, self._NUM_SPECIAL:, :]            # (1, P, D)
        patch_pos = patch_pos.reshape(1, orig_pt, orig_pf, D)
        patch_pos = patch_pos.permute(0, 3, 1, 2)           # (1, D, T, F)

        patch_pos = F.interpolate(
            patch_pos.float(),
            size=(new_pt, new_pf),
            mode="bilinear", align_corners=False,
        )                                                    # (1, D, T', F)

        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, D)
        new_pos = torch.cat([cls_dist, patch_pos], dim=1)

        self.ast.embeddings.position_embeddings = nn.Parameter(new_pos)
        log.info("AST pos-embeddings resized: %d → %d tokens "
                 "(T: %d → %d patches)",
                 pos.shape[1], new_pos.shape[1], orig_pt, new_pt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.ast(input_values=x)
        return outputs.pooler_output                     # (B, 768)

    def get_encoder_layers(self):
        return list(self.ast.encoder.layer)


#  Classifier Head  &  Transfer Model wrapper

class ClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = CLASSIFIER_HIDDEN,
                 num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class TransferModel(nn.Module):
    """Pretrained backbone + interval classification head.

    Provides convenience methods for freezing / unfreezing subsets of
    parameters according to the three fine-tuning strategies.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = ClassifierHead(embed_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_panns_partial(self,
                               last_n: int = PANNS_UNFREEZE_BLOCKS) -> None:
        self.freeze_backbone()
        for block in self.backbone.get_blocks()[-last_n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.fc1.parameters():
            p.requires_grad = True

    def unfreeze_ast_partial(self,
                             last_n: int = AST_UNFREEZE_LAYERS) -> None:
        self.freeze_backbone()
        for layer in self.backbone.get_encoder_layers()[-last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.backbone.ast.layernorm.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def backbone_params(self):
        return (p for p in self.backbone.parameters() if p.requires_grad)

    def head_params(self):
        return self.head.parameters()

    def train(self, mode: bool = True):
        self.training = mode
        self.head.train(mode)
        any_trainable = any(p.requires_grad
                            for p in self.backbone.parameters())
        if any_trainable:
            self.backbone.train(mode)
            if mode:
                self._eval_frozen_batchnorms()
        else:
            self.backbone.eval()
        return self

    def _eval_frozen_batchnorms(self) -> None:
        """Keep BN layers whose params are frozen in eval mode so their
        running statistics are not altered during fine-tuning."""
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()


#  Factory functions

def build_transfer_model(
    model_type: str,
    finetune_level: str,
    device: torch.device,
    dropout: float = DROPOUT,
) -> TransferModel:
    """Instantiate a pretrained backbone + classifier, apply the
    requested freeze strategy, and move to device."""
    if model_type == "panns":
        backbone = load_panns_backbone()
        embed_dim = PANNS_EMBED_DIM
    elif model_type == "ast":
        backbone = ASTBackbone()
        embed_dim = AST_EMBED_DIM
    else:
        raise ValueError(f"Unknown pretrained model type: {model_type}")

    model = TransferModel(backbone, embed_dim, dropout=dropout)

    if finetune_level == "frozen":
        model.freeze_backbone()
    elif finetune_level == "partial":
        if model_type == "panns":
            model.unfreeze_panns_partial()
        else:
            model.unfreeze_ast_partial()
    elif finetune_level == "full":
        model.unfreeze_all()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("%s [%s] — trainable: %d / %d (%.1f %%)",
             model_type, finetune_level, trainable, total,
             100.0 * trainable / total)
    return model.to(device)


def build_scratch_model(
    representation: str = "mel",
    dropout: float = DROPOUT,
) -> nn.Module:
    """Build from-scratch CNN for mel, CQT, or HCQT."""
    from experiments.exp1_representations.models import SpectrogramCNN
    in_ch = 5 if representation == "hcqt" else 1
    return SpectrogramCNN(in_channels=in_ch, dropout=dropout)
