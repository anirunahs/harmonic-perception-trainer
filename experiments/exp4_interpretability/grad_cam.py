"""
Grad-CAM for HCQT CNN: visualize which frequency-time regions the model attends to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import HCQT_PARAMS
from .models import HCQT_CNN


def _cqt_bin_to_hz(bin_index: np.ndarray, fmin: float, bins_per_octave: int) -> np.ndarray:
    """Convert CQT bin indices to Hz."""
    return fmin * (2.0 ** (bin_index / bins_per_octave))


def _grad_cam_weights_and_activation(
    model: HCQT_CNN,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Grad-CAM weights and activation."""
    model.eval()
    saved_act = []
    saved_grad = []

    def fwd_hook(module: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
        saved_act.append(out.detach())

    def bwd_hook(module: nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor) -> None:
        saved_grad.append(grad_out[0].detach())

    handle_fwd = model.features.register_forward_hook(fwd_hook)
    handle_bwd = model.features.register_full_backward_hook(bwd_hook)

    x = x.to(device)
    x.requires_grad_(True)
    logits = model(x)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    A = saved_act[0]
    grad = saved_grad[0]
    return A, grad


def compute_grad_cam(
    model: HCQT_CNN,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Compute Grad-CAM heatmap."""
    A, grad = _grad_cam_weights_and_activation(model, x, target_class=target_class, device=device)
    weights = grad.mean(dim=(2, 3))
    cam = (weights.unsqueeze(-1).unsqueeze(-1) * A).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    return cam.squeeze().cpu().numpy()


def overlay_heatmap_on_spectrogram(
    spec: np.ndarray,
    heatmap: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay heatmap on spectrogram."""
    if spec.ndim == 3:
        spec_2d = spec[0] if spec.shape[0] == 1 else spec.mean(axis=0)
    else:
        spec_2d = spec
    F, T = spec_2d.shape
    h = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    heatmap_resized = torch.nn.functional.interpolate(
        h, size=(F, T), mode="bilinear", align_corners=False
    ).squeeze().numpy()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(spec_2d, aspect="auto", origin="lower", cmap="gray")
    ax.imshow(heatmap_resized, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    return ax


def save_gradcam_figure(
    spec: np.ndarray,
    heatmap: np.ndarray,
    interval_name: str,
    pred_class: int,
    out_path: Path,
    title: Optional[str] = None,
    pred_name: Optional[str] = None,
    source: str = "synthetic",
) -> None:
    """Plot spectrogram, heatmap side-by-side and overlay for clarity. Y-axis in Hz."""
    p = HCQT_PARAMS
    fmin, n_bins, bpo = p["fmin"], p["n_bins"], p["bins_per_octave"]
    if spec.ndim == 3:
        spec_2d = spec[0]
    else:
        spec_2d = spec
    F, T = spec_2d.shape
    s = np.clip(spec_2d, np.percentile(spec_2d, 5), np.percentile(spec_2d, 98))
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    h_norm = np.maximum(heatmap, 0)
    if h_norm.max() > h_norm.min():
        h_norm = (h_norm - h_norm.min()) / (h_norm.max() - h_norm.min())
    h_resized = torch.nn.functional.interpolate(
        torch.from_numpy(h_norm).float().unsqueeze(0).unsqueeze(0),
        size=(F, T), mode="bilinear", align_corners=False
    ).squeeze().numpy()
    y_bins = np.arange(F)
    y_hz = _cqt_bin_to_hz(y_bins, fmin, bpo)
    extent = [0, T, y_hz[0], y_hz[-1]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    ax1.imshow(s, aspect="auto", origin="lower", extent=extent, cmap="gray_r")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_xlabel("Time (frames)")
    ax1.set_title("HCQT (1st harmonic)")

    ax2.imshow(h_resized, aspect="auto", origin="lower", extent=extent, cmap="jet", vmin=0, vmax=1)
    ax2.set_xlabel("Time (frames)")
    ax2.set_title("Grad-CAM (attention)")

    ax3.imshow(s, aspect="auto", origin="lower", extent=extent, cmap="gray_r")
    im = ax3.imshow(h_resized, aspect="auto", origin="lower", extent=extent, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    ax3.set_xlabel("Time (frames)")
    ax3.set_title("Overlay")
    plt.colorbar(im, ax=ax3, label="Attention", shrink=0.7)

    tit = title or f"True: {interval_name}  |  Pred: {pred_name or pred_class}  ({source})"
    fig.suptitle(tit, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
