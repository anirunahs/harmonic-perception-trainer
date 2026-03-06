"""
Evaluation and report formatting for Experiment 3.

Computes accuracy, macro-F1, Cohen's κ, per-class metrics, confusion matrix.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from .config import (
    EXP3_RESULTS_TXT,
    INTERVAL_CLASSES,
    NUM_CLASSES,
    RESULTS_DIR,
)

log = logging.getLogger(__name__)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict:
    acc = float(accuracy_score(targets, preds))
    f1_macro = float(f1_score(targets, preds, average="macro", zero_division=0))
    f1_weighted = float(f1_score(targets, preds, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(targets, preds))
    cm = confusion_matrix(targets, preds, labels=range(NUM_CLASSES))

    per_class = {}
    for i, cls in enumerate(INTERVAL_CLASSES):
        tp = cm[i, i]
        pred_pos = cm[:, i].sum()
        true_pos = cm[i, :].sum()
        prec = tp / pred_pos if pred_pos > 0 else 0.0
        rec = tp / true_pos if true_pos > 0 else 0.0
        f1_cls = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1_cls}

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "cohen_kappa": kappa,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def format_results(
    seed_metrics: List[Dict],
    split_sizes: Dict[str, int],
    histories: List["TrainHistory"],
    output_path: Path = EXP3_RESULTS_TXT,
    split_mode: str = "position",
    synthetic_metrics: Optional[List[Dict]] = None,
) -> str:
    """Format experiment 3 report and write to output_path."""
    from .trainer import TrainHistory

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(seed_metrics)
    accs = [m["accuracy"] for m in seed_metrics]
    f1s = [m["macro_f1"] for m in seed_metrics]
    kappas = [m["cohen_kappa"] for m in seed_metrics]

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.4f} ± {std:.4f}"

    split_desc = (
        "train/val = NF, test = MF (robustness to mic position)"
        if split_mode == "position"
        else "train = 6 classes NF, val = 2 NF, test = 4 held-out classes MF"
    )
    lines = [
        "=" * 80,
        "  EXPERIMENT 3 — FINE-TUNING ON RECORDED (MICROPHONE) DATASET",
        f"  Split: {split_desc}",
        "=" * 80,
        "",
        "RQ: Does a model (best HCQT from Exp 1) adapt to real microphone",
        "    recordings and retain accuracy?",
        "",
        "Recorded split:",
        f"  train (NF): {split_sizes.get('train', 0)} segments",
        f"  val   (NF): {split_sizes.get('val', 0)} segments",
        f"  test (MF): {split_sizes.get('test', 0)} segments",
        "",
        "RECORDED TEST (MF) ",
        f"  Accuracy:   {_fmt(float(np.mean(accs)), float(np.std(accs)))}",
        f"  Macro-F1:  {_fmt(float(np.mean(f1s)), float(np.std(f1s)))}",
        f"  Cohen κ:   {_fmt(float(np.mean(kappas)), float(np.std(kappas)))}",
        "",
        "TRAINING",
    ]
    if histories:
        best_epochs = [h.best_epoch for h in histories]
        total_times = [h.total_time_s for h in histories]
        lines.append(f"  Best epoch (val F1): {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}" if len(best_epochs) > 1 else f"  Best epoch (val F1): {best_epochs[0]}")
        lines.append(f"  Total time (s): {np.mean(total_times):.1f} ± {np.std(total_times):.1f}" if len(total_times) > 1 else f"  Total time (s): {total_times[0]:.1f}")
    lines.append("")

    if synthetic_metrics:
        s_acc = [m["accuracy"] for m in synthetic_metrics]
        s_f1 = [m["macro_f1"] for m in synthetic_metrics]
        lines.extend([
            "SYNTHETIC TEST (violin, flute) after fine-tune",
            f"  Accuracy:   {_fmt(float(np.mean(s_acc)), float(np.std(s_acc)))}",
            f"  Macro-F1:  {_fmt(float(np.mean(s_f1)), float(np.std(s_f1)))}",
            "",
        ])

    if seed_metrics and "per_class" in seed_metrics[0]:
        lines.append("PER-CLASS (test, averaged over seeds)")
        for cls in INTERVAL_CLASSES:
            precs = [m["per_class"][cls]["precision"] for m in seed_metrics]
            recs = [m["per_class"][cls]["recall"] for m in seed_metrics]
            f1cls = [m["per_class"][cls]["f1"] for m in seed_metrics]
            lines.append(
                f"  {cls:<14s}  P {np.mean(precs):.3f}  R {np.mean(recs):.3f}  F1 {np.mean(f1cls):.3f}"
            )
        lines.append("")

    if seed_metrics and "confusion_matrix" in seed_metrics[0]:
        cm_mean = np.mean([m["confusion_matrix"] for m in seed_metrics], axis=0)
        lines.append("CONFUSION MATRIX (test, mean)")
        lines.append("  " + " ".join(f"{c[:6]:>6}" for c in INTERVAL_CLASSES))
        for i, row in enumerate(cm_mean):
            lines.append("  " + " ".join(f"{v:6.1f}" for v in row) + f"  {INTERVAL_CLASSES[i]}")
        lines.append("")

    lines.extend(["END OF EXPERIMENT 3 RESULTS"])
    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    log.info("Results written to %s", output_path)
    return report


def format_results_comparison(
    results_by_repr: Dict[str, List[Dict]],
    histories_by_repr: Dict[str, List["TrainHistory"]],
    split_sizes: Dict[str, int],
    output_path: Path = EXP3_RESULTS_TXT,
    split_mode: str = "position",
) -> str:
    """Format Exp 3 report with comparison of hcqt / mel / cqt / hcqt_scratch."""
    from .trainer import TrainHistory

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.4f} ± {std:.4f}"

    split_desc = (
        "train/val = NF, test = MF (robustness to mic position)"
        if split_mode == "position"
        else "train = 6 classes NF, val = 2 NF, test = 4 held-out classes MF"
    )
    lines = [
        "=" * 80,
        "  EXPERIMENT 3 — FINE-TUNING ON RECORDED (MICROPHONE) DATASET",
        f"  Split: {split_desc}",
        "=" * 80,
        "",
        "Comparison: HCQT zero-shot (no fine-tune), HCQT/Mel/CQT (pretrained+fine-tune), HCQT from scratch (recorded only).",
        "",
        "Recorded split:",
        f"  train (NF): {split_sizes.get('train', 0)} segments",
        f"  val   (NF): {split_sizes.get('val', 0)} segments",
        f"  test (MF): {split_sizes.get('test', 0)} segments",
        "",
        "--- COMPARISON (RECORDED TEST MF, mean ± std over seeds) ---",
        "",
        "  Representation     Accuracy         Macro-F1        Cohen κ",
        "  " + "-" * 60,
    ]
    repr_order = ("hcqt_zero_shot", "hcqt", "mel", "cqt", "hcqt_scratch")
    for repr_name in repr_order:
        if repr_name not in results_by_repr or not results_by_repr[repr_name]:
            continue
        metrics_list = results_by_repr[repr_name]
        accs = [m["accuracy"] for m in metrics_list]
        f1s = [m["macro_f1"] for m in metrics_list]
        kappas = [m["cohen_kappa"] for m in metrics_list]
        if repr_name == "hcqt_zero_shot":
            label = "HCQT (zero-shot)"
        elif repr_name == "hcqt_scratch":
            label = "HCQT (scratch)"
        else:
            label = repr_name.upper()
        lines.append(
            f"  {label:<18}  {_fmt(float(np.mean(accs)), float(np.std(accs)))}  "
            f"{_fmt(float(np.mean(f1s)), float(np.std(f1s)))}  "
            f"{_fmt(float(np.mean(kappas)), float(np.std(kappas)))}"
        )
    lines.append("")

    lines.append("TRAINING (best epoch, total time s)")
    for repr_name in ("hcqt", "mel", "cqt", "hcqt_scratch"):
        if repr_name not in histories_by_repr or not histories_by_repr[repr_name]:
            continue
        hist = histories_by_repr[repr_name]
        best_epochs = [h.best_epoch for h in hist]
        times = [h.total_time_s for h in hist]
        label = "HCQT (scratch)" if repr_name == "hcqt_scratch" else repr_name.upper()
        lines.append(
            f"  {label:<18}  best_epoch: {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}  "
            f"time_s: {np.mean(times):.1f} ± {np.std(times):.1f}"
        )
    lines.append("")
    lines.append("END OF EXPERIMENT 3 RESULTS")
    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    log.info("Results written to %s", output_path)
    return report
