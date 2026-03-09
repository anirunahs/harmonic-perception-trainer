"""
t-SNE, UMAP, and confusion matrix visualization for Experiment 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from .config import INTERVAL_CLASSES, IDX_TO_LABEL, TSNE_PERPLEXITY, UMAP_N_NEIGHBORS, UMAP_MIN_DIST

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def sanitize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Replace nan/inf and avoid zero variance to prevent t-SNE/PCA divide-by-zero."""
    out = np.asarray(embeddings, dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    for j in range(out.shape[1]):
        if np.var(out[:, j]) < 1e-12:
            out[:, j] = out[:, j] + np.random.RandomState(42).randn(out.shape[0]) * 1e-8
    return out.astype(np.float32)


def run_tsne(embeddings: np.ndarray, perplexity: int = TSNE_PERPLEXITY, seed: int = 42) -> np.ndarray:
    """2D t-SNE projection."""
    embeddings = sanitize_embeddings(embeddings)
    perplexity = min(perplexity, max(2, len(embeddings) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
    return tsne.fit_transform(embeddings)


def run_umap(embeddings: np.ndarray, n_neighbors: int = UMAP_N_NEIGHBORS, min_dist: float = UMAP_MIN_DIST) -> np.ndarray:
    """2D UMAP projection."""
    if not HAS_UMAP:
        raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
    embeddings = sanitize_embeddings(embeddings)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)


INTERVAL_COLORS_12 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]


def plot_embedding_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str = "Embedding",
    marker_by: Optional[np.ndarray] = None,
    label_names: Optional[list] = None,
    marker_labels: Optional[list] = None,
) -> None:
    """Plot 2D embeddings. One color per interval; circle = synthetic, square = recorded."""
    n_classes = max(labels) + 1
    label_names = label_names or [IDX_TO_LABEL.get(i, str(i)) for i in range(n_classes)]
    colors = INTERVAL_COLORS_12[:n_classes] if n_classes <= 12 else plt.cm.tab20(np.linspace(0, 1, n_classes))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    legend_handles = []
    if marker_by is not None:
        groups = np.unique(marker_by)
        markers = ["o", "s"]
        for c in range(n_classes):
            added_legend = False
            for gi, g in enumerate(groups):
                m = (marker_by == g) & (labels == c)
                if m.sum() == 0:
                    continue
                color = colors[c] if isinstance(colors, list) else colors[c]
                sc = ax.scatter(
                    coords[m, 0], coords[m, 1],
                    c=[color], marker=markers[gi % len(markers)], alpha=0.7, s=28, edgecolors="white", linewidths=0.3
                )
                if not added_legend:
                    from matplotlib.lines import Line2D
                    legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                                                markeredgecolor="gray", markersize=10, label=label_names[c]))
                    added_legend = True
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="k",
                                      markersize=10, label="synthetic"))
        legend_handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="k",
                                      markersize=10, label="recorded"))
    else:
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            color = colors[c] if isinstance(colors, list) else colors[c]
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], label=label_names[c], alpha=0.7, s=28)
    if legend_handles:
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str = "Confusion matrix",
    class_names: Optional[list] = None,
    normalize: Optional[str] = "true",
) -> None:
    """Plot and save confusion matrix."""
    class_names = class_names or INTERVAL_CLASSES
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize == "true":
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    elif normalize == "pred":
        cm = cm.astype(float) / (cm.sum(axis=0, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    im = ax.imshow(cm, aspect="auto", cmap="Blues", vmin=0, vmax=1.0 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else str(int(v)), ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
