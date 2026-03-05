"""
Comprehensive evaluation and results formatting for Experiment 2.

Computes per-split and per-class metrics, confusion matrices, bias /
variance / mismatch gap analysis, and writes a detailed results text
file.
"""

from __future__ import annotations

import datetime
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from .config import (
    AST_EMBED_DIM,
    IDX_TO_LABEL,
    INTERVAL_CLASSES,
    NUM_CLASSES,
    PANNS_EMBED_DIM,
    USE_AMP,
)
from .trainer import TrainHistory

log = logging.getLogger(__name__)


#  Core metric computation

@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_all, tgts_all = [], []
    use_amp = USE_AMP and device.type == "cuda"
    for x, y in loader:
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(x.to(device))
        preds_all.append(logits.argmax(1).cpu().numpy())
        tgts_all.append(y.numpy())
    return np.concatenate(preds_all), np.concatenate(tgts_all)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict:
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(targets, preds)

    prec, rec, f1_cls, sup = precision_recall_fscore_support(
        targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
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
    results = {}
    for name, loader in loaders.items():
        preds, targets = predict(model, loader, device)
        results[name] = compute_metrics(preds, targets)
        log.info("  %s — acc %.3f  f1 %.3f  κ %.3f",
                 name, results[name]["accuracy"],
                 results[name]["macro_f1"],
                 results[name]["cohen_kappa"])
    return results


#  Save / load per-condition results (for launcher merge)

def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return obj


def save_condition_results(
    cond_name: str,
    agg: Dict[str, Dict],
    info: Dict,
    split_sizes: Dict[str, int],
    histories: List[TrainHistory],
    path: Path,
) -> None:
    """Save one condition's results to JSON for later merge."""
    payload = {
        "cond_name": cond_name,
        "all_cond_results": {cond_name: _to_serializable(agg)},
        "condition_info": {cond_name: info},
        "split_sizes": split_sizes,
        "histories": [
            {"best_epoch": h.best_epoch, "total_time_s": h.total_time_s}
            for h in histories
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_and_merge_condition_results(
    results_dir: Path,
    condition_names: List[str],
) -> Tuple[Dict, Dict[str, List[TrainHistory]], Dict, Dict[str, int]]:
    """Load JSON files for each condition and merge into full structures."""
    all_cond_results: Dict[str, Dict] = {}
    condition_info: Dict[str, Dict] = {}
    split_sizes: Dict[str, int] = {}
    all_histories: Dict[str, List[TrainHistory]] = {}

    for name in condition_names:
        path = results_dir / f"exp2_{name}_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        all_cond_results.update(data["all_cond_results"])
        condition_info.update(data["condition_info"])
        split_sizes.update(data["split_sizes"])

        all_histories[name] = [
            TrainHistory(best_epoch=h["best_epoch"], total_time_s=h["total_time_s"])
            for h in data["histories"]
        ]

    for cond_name, agg in all_cond_results.items():
        for split in agg:
            if "confusion_matrix_mean" in agg[split]:
                agg[split]["confusion_matrix_mean"] = np.array(
                    agg[split]["confusion_matrix_mean"]
                )

    return all_cond_results, all_histories, condition_info, split_sizes


#  Aggregation across seeds

def aggregate_seeds(seed_results: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    splits = seed_results[0].keys()
    agg: Dict[str, Dict] = {}
    for split in splits:
        agg[split] = {}
        for mk in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa"):
            vals = [r[split][mk] for r in seed_results]
            agg[split][mk] = {"mean": float(np.mean(vals)),
                              "std": float(np.std(vals))}
        all_cm = np.stack(
            [r[split]["confusion_matrix"] for r in seed_results])
        agg[split]["confusion_matrix_mean"] = np.mean(all_cm, axis=0)

        agg[split]["per_class"] = {}
        for cls in INTERVAL_CLASSES:
            agg[split]["per_class"][cls] = {}
            for pm in ("precision", "recall", "f1"):
                vals = [r[split]["per_class"][cls][pm]
                        for r in seed_results]
                agg[split]["per_class"][cls][pm] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
    return agg


#  Error-gap analysis

def error_gap_analysis(agg: Dict[str, Dict]) -> Dict[str, float]:
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


#  Text formatting

_W = 88


def _h1(title: str) -> str:
    return f"\n{'=' * _W}\n  {title}\n{'=' * _W}\n"


def _h2(title: str) -> str:
    return f"\n{'─' * _W}\n  {title}\n{'─' * _W}\n"


def _fmt(val: Dict) -> str:
    return f"{val['mean']:.4f} ± {val['std']:.4f}"


def _format_cm(cm: np.ndarray, labels: list[str]) -> str:
    short = [l[:6] for l in labels]
    lines = ["        " + " ".join(f"{s:>6s}" for s in short)]
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:6.1f}" for v in row)
        lines.append(f"{short[i]:>7s} {row_str}")
    return "\n".join(lines) + "\n"


def format_results(
    all_cond_results: Dict[str, Dict[str, Dict]],
    all_histories: Dict[str, List[TrainHistory]],
    condition_info: Dict[str, Dict],
    split_sizes: Dict[str, int],
    exp1_reference: Dict[str, str] | None = None,
) -> str:
    buf = StringIO()

    # Header
    buf.write("=" * _W + "\n")
    buf.write("  EXPERIMENT 2 — TRANSFER LEARNING FROM PRETRAINED "
              "AUDIO MODELS\n")
    buf.write("  Musical Interval Recognition\n")
    buf.write("=" * _W + "\n\n")
    buf.write(f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    buf.write(f"Conditions: {', '.join(all_cond_results.keys())}\n\n")
    buf.write(
        "RQ: How effective is transfer learning from general-purpose\n"
        "    audio models (PANNs CNN14, AST) for the specialised task\n"
        "    (A) From-scratch with mel/CQT/HCQT vs (B) transfer from\n"
        "    general-purpose audio models; how does fine-tuning depth affect\n"
        "    transfer performance?\n"
    )

    buf.write(_h1("1. DATASET"))
    buf.write(
        "Synthetic multi-timbral dataset, 10 instruments, "
        "12 interval classes.\n"
        "Split by instrument (6 / 2 / 2).\n\n"
    )
    for name, n in split_sizes.items():
        buf.write(f"  {name:>10s}: {n:5d} samples\n")
    buf.write(f"\n  Total: {sum(split_sizes.values()):>10d}\n")

    buf.write(_h1("2. PRETRAINED MODELS"))
    buf.write(
        "  PANNs CNN14\n"
        "    Pretrained on: AudioSet (1.9 M clips, 527 classes)\n"
        f"    Parameters:    ~80 M\n"
        f"    Embedding dim: {PANNS_EMBED_DIM}\n"
        "    Input:         log₁₀-mel spectrogram, 64 bands, 32 kHz\n\n"
        "  AST — Audio Spectrogram Transformer (Gong et al., 2021)\n"
        "    Pretrained on: AudioSet (2 M clips, 527 classes)\n"
        f"    Parameters:    ~87 M (ViT-Base)\n"
        f"    Embedding dim: {AST_EMBED_DIM}\n"
        "    Input:         normalised fbank, 128 bands, 16 kHz\n\n"
        "  From-scratch CNNs\n"
        "    Mel:  ~1.6 M params, log-mel 128 bands, 22 050 Hz\n"
        "    CQT:  same CNN, log-CQT (1, 252, T), 22 050 Hz\n"
        "    HCQT: same CNN, log-HCQT (5, 252, T), 22 050 Hz\n"
    )

    buf.write(_h1("3. CONDITIONS AND HYPERPARAMETERS"))
    buf.write(
        "  Fine-tuning levels:\n"
        "    frozen  — backbone frozen, only classifier head trained\n"
        "    partial — last 2 blocks / layers + head trainable\n"
        "    full    — all parameters trainable, differential LR\n\n"
    )
    for cond_name, info in condition_info.items():
        buf.write(f"  [{cond_name}]\n")
        for k, v in info.items():
            buf.write(f"    {k:<25s}: {v}\n")
        buf.write("\n")

    buf.write(_h1("4. MAIN RESULTS"))
    hdr = (f"  {'Condition':<20s} {'Accuracy':>16s} {'Macro-F1':>16s} "
           f"{'Weighted-F1':>16s} {'Cohen κ':>16s}")
    buf.write(hdr + "\n  " + "─" * 86 + "\n")

    for cond_name, agg in all_cond_results.items():
        t = agg["test"]
        buf.write(
            f"  {cond_name:<20s} {_fmt(t['accuracy']):>16s} "
            f"{_fmt(t['macro_f1']):>16s} "
            f"{_fmt(t['weighted_f1']):>16s} "
            f"{_fmt(t['cohen_kappa']):>16s}\n"
        )

    if exp1_reference:
        buf.write("\n  Reference from Experiment 1 (same dataset & splits):\n")
        for k, v in exp1_reference.items():
            buf.write(f"    {k}: {v}\n")

    buf.write(_h1("5. ERROR-GAP ANALYSIS (bias / variance / mismatch)"))
    buf.write(
        "train vs train-dev vs val vs test\n\n"
    )
    ghdr = (f"  {'Condition':<20s} {'Train':>8s} {'T-Dev':>8s} "
            f"{'Val':>8s} {'Test':>8s} │ "
            f"{'Var':>7s} {'Mis':>7s} {'OT':>7s}")
    buf.write(ghdr + "\n  " + "─" * 86 + "\n")
    for cond_name, agg in all_cond_results.items():
        g = error_gap_analysis(agg)
        buf.write(
            f"  {cond_name:<20s} {g['train_error']:8.4f} "
            f"{g['train_dev_error']:8.4f} "
            f"{g['val_error']:8.4f} "
            f"{g['test_error']:8.4f} │ "
            f"{g['variance_gap']:7.4f} "
            f"{g['mismatch_gap']:7.4f} "
            f"{g['overtuning_gap']:7.4f}\n"
        )

    buf.write(_h1("6. PER-CLASS RESULTS ON TEST SET"))
    for cond_name, agg in all_cond_results.items():
        buf.write(_h2(f"{cond_name} — per-class (test)"))
        buf.write(f"  {'Interval':<14s} {'Precision':>16s} "
                  f"{'Recall':>16s} {'F1':>16s}\n")
        buf.write("  " + "─" * 64 + "\n")
        for cls in INTERVAL_CLASSES:
            pc = agg["test"]["per_class"][cls]
            buf.write(
                f"  {cls:<14s} "
                f"{pc['precision']['mean']:7.3f}±"
                f"{pc['precision']['std']:.3f} "
                f"{pc['recall']['mean']:7.3f}±"
                f"{pc['recall']['std']:.3f} "
                f"{pc['f1']['mean']:7.3f}±"
                f"{pc['f1']['std']:.3f}\n"
            )

    buf.write(_h1("7. CONFUSION MATRICES (test, averaged over seeds)"))
    for cond_name, agg in all_cond_results.items():
        buf.write(_h2(f"{cond_name} — test confusion matrix"))
        buf.write(_format_cm(
            agg["test"]["confusion_matrix_mean"], INTERVAL_CLASSES))

    buf.write(_h1("8. TRAINING SUMMARY"))
    for cond_name, histories in all_histories.items():
        ep = [h.best_epoch for h in histories]
        ts = [h.total_time_s for h in histories]
        buf.write(
            f"  {cond_name:<20s}  best_epoch: {np.mean(ep):5.1f}±"
            f"{np.std(ep):.1f}    "
            f"time: {np.mean(ts):7.1f}±{np.std(ts):.1f} s\n"
        )

    buf.write(_h1("9. KEY FINDINGS"))

    best_cond = max(
        all_cond_results.items(),
        key=lambda x: x[1]["test"]["macro_f1"]["mean"],
    )
    buf.write(f"  Best condition overall: {best_cond[0]}\n")
    buf.write(f"    Test accuracy:  {_fmt(best_cond[1]['test']['accuracy'])}\n")
    buf.write(f"    Test macro-F1:  {_fmt(best_cond[1]['test']['macro_f1'])}\n")
    buf.write("\n")

    for model_type in ("panns", "ast"):
        buf.write(f"  {model_type.upper()} fine-tuning progression:\n")
        for level in ("frozen", "partial", "full"):
            cn = f"{model_type}_{level}"
            if cn in all_cond_results:
                a = all_cond_results[cn]
                buf.write(
                    f"    {level:<10s}: acc {_fmt(a['test']['accuracy'])}, "
                    f"f1 {_fmt(a['test']['macro_f1'])}\n"
                )
        buf.write("\n")

    for scratch_name in ("cnn_mel_scratch", "cnn_cqt_scratch", "cnn_hcqt_scratch"):
        if scratch_name in all_cond_results:
            a = all_cond_results[scratch_name]
            buf.write(
                f"  {scratch_name}: acc {_fmt(a['test']['accuracy'])}, "
                f"f1 {_fmt(a['test']['macro_f1'])}\n"
            )
    buf.write("\n")


    buf.write("\n" + "=" * _W + "\n")
    buf.write("  END OF EXPERIMENT 2 RESULTS\n")
    buf.write("=" * _W + "\n")
    return buf.getvalue()
