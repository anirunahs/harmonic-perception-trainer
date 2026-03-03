"""
Comprehensive evaluation and results formatting.

Computes per-split and per-class metrics, confusion matrices,
bias / variance / mismatch gap analysis, and writes a single
results text file suitable for inclusion in the article.
"""

from __future__ import annotations

import datetime
import logging
from io import StringIO
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .config import IDX_TO_LABEL, INTERVAL_CLASSES, NUM_CLASSES
from .trainer import TrainHistory

log = logging.getLogger(__name__)

# Core metric computation


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_all, tgts_all = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds_all.append(logits.argmax(1).cpu().numpy())
        tgts_all.append(y.numpy())
    return np.concatenate(preds_all), np.concatenate(tgts_all)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict:
    """Return a dict with all requested evaluation metrics."""
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(targets, preds)

    prec, rec, f1_cls, sup = precision_recall_fscore_support(
        targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))

    per_class = {}
    for i, name in IDX_TO_LABEL.items():
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1_cls[i]),
            "support": int(sup[i]),
        }

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "cohen_kappa": float(kappa),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def evaluate_all_splits(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Dict]:
    """Evaluate a model on every split; return {split: metrics}."""
    results = {}
    for name, loader in loaders.items():
        preds, targets = predict(model, loader, device)
        results[name] = compute_metrics(preds, targets)
        log.info("  %s — acc %.3f  f1 %.3f  κ %.3f",
                 name, results[name]["accuracy"],
                 results[name]["macro_f1"],
                 results[name]["cohen_kappa"])
    return results


# Aggregation across seeds


def aggregate_seeds(
    seed_results: List[Dict[str, Dict]],
) -> Dict[str, Dict]:
    """Compute mean ± std of scalar metrics across seed runs."""
    splits = seed_results[0].keys()
    agg: Dict[str, Dict] = {}

    for split in splits:
        metrics_keys = ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa"]
        agg[split] = {}
        for mk in metrics_keys:
            vals = [r[split][mk] for r in seed_results]
            agg[split][mk] = {"mean": float(np.mean(vals)),
                              "std": float(np.std(vals))}

        all_cm = np.stack([r[split]["confusion_matrix"] for r in seed_results])
        agg[split]["confusion_matrix_mean"] = np.mean(all_cm, axis=0)

        agg[split]["per_class"] = {}
        for cls in INTERVAL_CLASSES:
            agg[split]["per_class"][cls] = {}
            for pm in ("precision", "recall", "f1"):
                vals = [r[split]["per_class"][cls][pm] for r in seed_results]
                agg[split]["per_class"][cls][pm] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    return agg


# Error-gap analysis


def error_gap_analysis(agg: Dict[str, Dict]) -> Dict[str, float]:
    """Compute bias / variance / mismatch gaps from aggregated metrics.

    train_error       – measures bias (underfitting)
    train-dev − train – measures variance (overfitting to training data)
    val − train-dev   – measures data mismatch (cross-timbral generalisation)
    test − val        – measures over-tuning to validation set
    """
    te = 1.0 - agg["train"]["accuracy"]["mean"]
    tde = 1.0 - agg["train_dev"]["accuracy"]["mean"]
    ve = 1.0 - agg["val"]["accuracy"]["mean"]
    tste = 1.0 - agg["test"]["accuracy"]["mean"]

    return {
        "train_error": te,
        "train_dev_error": tde,
        "val_error": ve,
        "test_error": tste,
        "variance_gap": tde - te,
        "mismatch_gap": ve - tde,
        "overtuning_gap": tste - ve,
    }


# Text formatting

_W = 80


def _h1(title: str) -> str:
    return f"\n{'=' * _W}\n  {title}\n{'=' * _W}\n"


def _h2(title: str) -> str:
    return f"\n{'─' * _W}\n  {title}\n{'─' * _W}\n"


def _fmt(val: Dict) -> str:
    return f"{val['mean']:.4f} ± {val['std']:.4f}"


def format_confusion_matrix(cm: np.ndarray, labels: List[str]) -> str:
    short = [l[:6] for l in labels]
    buf = StringIO()
    header = "        " + " ".join(f"{s:>6s}" for s in short)
    buf.write(header + "\n")
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:6.1f}" for v in row)
        buf.write(f"{short[i]:>7s} {row_str}\n")
    return buf.getvalue()


def format_results(
    all_repr_results: Dict[str, Dict[str, Dict]],
    all_histories: Dict[str, List[TrainHistory]],
    hyperparams: Dict[str, Dict],
    split_sizes: Dict[str, int],
) -> str:
    """Build the complete results text document."""
    buf = StringIO()

    # Header
    buf.write("=" * _W + "\n")
    buf.write("  EXPERIMENT 1 — TIME-FREQUENCY REPRESENTATION COMPARISON\n")
    buf.write("  Musical Interval Recognition\n")
    buf.write("=" * _W + "\n\n")
    buf.write(f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    buf.write(f"Representations compared: {', '.join(all_repr_results.keys())}\n")

    # Dataset summary
    buf.write(_h1("1. DATASET"))
    buf.write("Synthetic multi-timbral dataset, 10 instruments, 12 interval classes.\n")
    buf.write("Split by instrument (6 / 2 / 2):\n\n")
    for name, n in split_sizes.items():
        buf.write(f"  {name:>10s}: {n:5d} samples\n")
    buf.write(f"\n  Total: {sum(split_sizes.values()):>10d}\n")
    buf.write("\nTrain instruments : piano, guitar_nylon, guitar_steel, "
              "cello, church_organ, trumpet\n")
    buf.write("Val instruments   : clarinet, marimba\n")
    buf.write("Test instruments  : violin, flute\n")
    buf.write("Train-dev         : 15 % held out from train "
              "(same instrument distribution)\n")

    # Hyper-parameters
    buf.write(_h1("2. HYPER-PARAMETERS"))
    for repr_name, hp in hyperparams.items():
        buf.write(f"\n  [{repr_name.upper()}]\n")
        for k, v in hp.items():
            buf.write(f"    {k:<20s}: {v}\n")

    # Representation parameters
    buf.write(_h1("3. REPRESENTATION PARAMETERS"))
    from .config import MEL_PARAMS, CQT_PARAMS, HCQT_PARAMS, RAW_PARAMS
    for name, params in [("Mel", MEL_PARAMS), ("CQT", CQT_PARAMS),
                         ("HCQT", HCQT_PARAMS), ("Raw", RAW_PARAMS)]:
        buf.write(f"\n  [{name}]\n")
        for k, v in params.items():
            buf.write(f"    {k:<20s}: {v}\n")

    # Main results table
    buf.write(_h1("4. MAIN RESULTS (mean ± std over seeds)"))
    header = (f"  {'Repr':<8s} {'Accuracy':>16s} {'Macro-F1':>16s} "
              f"{'Weighted-F1':>16s} {'Cohen κ':>16s}")
    buf.write(header + "\n")
    buf.write("  " + "─" * 74 + "\n")
    for rn, agg in all_repr_results.items():
        t = agg["test"]
        buf.write(
            f"  {rn:<8s} {_fmt(t['accuracy']):>16s} "
            f"{_fmt(t['macro_f1']):>16s} "
            f"{_fmt(t['weighted_f1']):>16s} "
            f"{_fmt(t['cohen_kappa']):>16s}\n"
        )

    # Error-gap analysis
    buf.write(_h1("5. ERROR-GAP ANALYSIS (bias / variance / mismatch)"))
    buf.write("  Framework: Andrew Ng — train vs train-dev vs dev vs test\n\n")
    header_g = (f"  {'Repr':<8s} {'Train err':>10s} {'T-Dev err':>10s} "
                f"{'Val err':>10s} {'Test err':>10s} │ "
                f"{'Var gap':>8s} {'Mis gap':>8s} {'OT gap':>8s}")
    buf.write(header_g + "\n")
    buf.write("  " + "─" * 86 + "\n")
    for rn, agg in all_repr_results.items():
        gaps = error_gap_analysis(agg)
        buf.write(
            f"  {rn:<8s} {gaps['train_error']:10.4f} "
            f"{gaps['train_dev_error']:10.4f} "
            f"{gaps['val_error']:10.4f} "
            f"{gaps['test_error']:10.4f} │ "
            f"{gaps['variance_gap']:8.4f} "
            f"{gaps['mismatch_gap']:8.4f} "
            f"{gaps['overtuning_gap']:8.4f}\n"
        )
    buf.write(
        "\n  Var gap  = train_dev_err − train_err    "
        "(high → overfitting to training data)\n"
        "  Mis gap  = val_err − train_dev_err       "
        "(high → poor cross-timbral generalisation)\n"
        "  OT gap   = test_err − val_err            "
        "(high → over-tuned to validation set)\n"
    )

    # Per-class results for each representation (test split)
    buf.write(_h1("6. PER-CLASS RESULTS ON TEST SET"))
    for rn, agg in all_repr_results.items():
        buf.write(_h2(f"{rn.upper()} — per-class (test)"))
        buf.write(f"  {'Interval':<14s} {'Precision':>16s} "
                  f"{'Recall':>16s} {'F1':>16s}\n")
        buf.write("  " + "─" * 64 + "\n")
        for cls in INTERVAL_CLASSES:
            pc = agg["test"]["per_class"][cls]
            buf.write(
                f"  {cls:<14s} "
                f"{pc['precision']['mean']:7.3f}±{pc['precision']['std']:.3f} "
                f"{pc['recall']['mean']:7.3f}±{pc['recall']['std']:.3f} "
                f"{pc['f1']['mean']:7.3f}±{pc['f1']['std']:.3f}\n"
            )

    # Confusion matrices (test, averaged over seeds)
    buf.write(_h1("7. CONFUSION MATRICES (test, averaged over seeds)"))
    for rn, agg in all_repr_results.items():
        buf.write(_h2(f"{rn.upper()} — test confusion matrix"))
        buf.write(format_confusion_matrix(
            agg["test"]["confusion_matrix_mean"], INTERVAL_CLASSES))

    # Training curves summary
    buf.write(_h1("8. TRAINING SUMMARY"))
    for rn, histories in all_histories.items():
        epochs_list = [h.best_epoch for h in histories]
        times_list = [h.total_time_s for h in histories]
        buf.write(
            f"  {rn:<8s}  best_epoch: {np.mean(epochs_list):5.1f}±"
            f"{np.std(epochs_list):.1f}    "
            f"time: {np.mean(times_list):6.1f}±{np.std(times_list):.1f} s\n"
        )

    buf.write("\n" + "=" * _W + "\n")
    buf.write("  END OF EXPERIMENT 1 RESULTS\n")
    buf.write("=" * _W + "\n")
    return buf.getvalue()
