"""
Experiment 4: Interpretability (Grad-CAM, t-SNE/UMAP, confusion matrix) and FFT-MLP baseline.

Run from project root: python -m experiments.exp4_interpretability.run [--use-exp1] [--seed 42]
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings
warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*empty frequency set.*", category=UserWarning)

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from .config import (
    SEED,
    RESULTS_DIR,
    EXP4_RESULTS_TXT,
    EXP4_RUN_LOG,
    EXP4_FIGURES_DIR,
    EXP4_GRADCAM_DIR,
    BEST_HCQT_EXP1,
    BEST_HCQT_EXP3,
    INTERVAL_CLASSES,
    NUM_CLASSES,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    GRADCAM_N_SAMPLES_PER_CLASS,
    BATCH_SIZE,
    MAX_EPOCHS,
    LR,
    WEIGHT_DECAY,
    PATIENCE,
    FFT_LABEL_SMOOTHING,
    FFT_LR_T_MAX,
    SYNTHETIC_METADATA,
    SYNTHETIC_DIR,
    RECORDED_METADATA,
    RECORDED_DIR,
    RECORDED_VAL_FRACTION,
    RECORDED_SPLIT_BY_FILE,
    TRAIN_DEV_FRACTION,
)
from .dataset import (
    get_synthetic_hcqt_splits,
    get_synthetic_hcqt_test_loader,
    get_recorded_hcqt_loaders,
    extract_fft_splits_synthetic,
    extract_fft_splits_recorded,
    create_fft_dataloaders,
)
from .models import build_hcqt_cnn, load_hcqt_checkpoint, build_fft_mlp
from .grad_cam import compute_grad_cam, save_gradcam_figure
from .visualization import run_tsne, run_umap, plot_embedding_2d, plot_confusion_matrix
from .fft_features import get_fft_feature_dim

log = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


@torch.no_grad()
def get_embeddings_and_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return embeddings (n, dim), predictions (n,), targets (n,)."""
    model.eval()
    embs, preds, targets = [], [], []
    for x, y in loader:
        x = x.to(device)
        emb = model.forward_embeddings(x)
        out = model(x)
        embs.append(emb.cpu().numpy())
        preds.append(out.argmax(1).cpu().numpy())
        targets.append(y.numpy())
    return (
        np.concatenate(embs),
        np.concatenate(preds),
        np.concatenate(targets),
    )


def run_gradcam(
    model: torch.nn.Module,
    device: torch.device,
    synthetic_splits: dict,
    out_dir: Path,
    n_per_class: int = GRADCAM_N_SAMPLES_PER_CLASS,
) -> None:
    """Generate Grad-CAM for a few synthetic test samples per class."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feats, labels = synthetic_splits["test"]
    rng = np.random.default_rng(SEED)
    for c in range(NUM_CLASSES):
        idx = np.where(np.array(labels) == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        for i, sample_idx in enumerate(chosen):
            x = torch.from_numpy(feats[sample_idx:sample_idx + 1]).float().to(device)
            spec = feats[sample_idx]
            pred = model(x).argmax(1).item()
            heatmap = compute_grad_cam(model, x, target_class=pred, device=device)
            name = INTERVAL_CLASSES[c]
            save_gradcam_figure(
                spec,
                heatmap,
                name,
                pred,
                out_dir / f"gradcam_synthetic_{name}_{i}.png",
                pred_name=IDX_TO_LABEL.get(pred, str(pred)),
                source="synthetic",
            )
    log.info("Grad-CAM (synthetic) figures saved to %s", out_dir)


def run_gradcam_recorded(
    model: torch.nn.Module,
    device: torch.device,
    recorded_test_loader: DataLoader,
    out_dir: Path,
    n_per_class: int = GRADCAM_N_SAMPLES_PER_CLASS,
) -> None:
    """Generate Grad-CAM for a few recorded test samples per class."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_class = {c: [] for c in range(NUM_CLASSES)}
    for x, y in recorded_test_loader:
        for i in range(y.size(0)):
            c = y[i].item()
            if len(by_class[c]) < n_per_class:
                by_class[c].append((x[i : i + 1], x[i].cpu().numpy()))
    rng = np.random.default_rng(SEED)
    for c in range(NUM_CLASSES):
        if not by_class[c]:
            continue
        pool = by_class[c]
        chosen = pool if len(pool) <= n_per_class else [pool[j] for j in rng.choice(len(pool), size=n_per_class, replace=False)]
        for i, (x_batch, spec) in enumerate(chosen):
            x_batch = x_batch.float().to(device)
            pred = model(x_batch).argmax(1).item()
            heatmap = compute_grad_cam(model, x_batch, target_class=pred, device=device)
            name = INTERVAL_CLASSES[c]
            save_gradcam_figure(
                spec,
                heatmap,
                name,
                pred,
                out_dir / f"gradcam_recorded_{name}_{i}.png",
                pred_name=IDX_TO_LABEL.get(pred, str(pred)),
                source="recorded",
            )
    log.info("Grad-CAM (recorded) figures saved to %s", out_dir)


def get_fft_embeddings_and_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return embeddings (n, dim), predictions (n,), targets (n,) for FFT MLP."""
    model.eval()
    embs, preds, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            embs.append(model.forward_embeddings(x).cpu().numpy())
            preds.append(model(x).argmax(1).cpu().numpy())
            targets.append(y.numpy())
    return (
        np.concatenate(embs),
        np.concatenate(preds),
        np.concatenate(targets),
    )


def train_fft_mlp(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    t_max: int = FFT_LR_T_MAX,
) -> tuple[torch.nn.Module, float]:
    """Train FFT MLP with cosine LR schedule. Return model and best val accuracy."""
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=t_max)
    criterion = nn.CrossEntropyLoss(label_smoothing=FFT_LABEL_SMOOTHING)
    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    for ep in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        scheduler.step()
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                val_preds.append(model(x).argmax(1).cpu().numpy())
                val_targets.append(y.numpy())
        val_acc = float(accuracy_score(np.concatenate(val_targets), np.concatenate(val_preds)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            log.info("Early stop at epoch %d", ep + 1)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device), best_val_acc


def evaluate_fft_mlp(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.append(model(x).argmax(1).cpu().numpy())
            targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(targets, preds)),
        "preds": preds,
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: Interpretability and FFT baseline")
    parser.add_argument("--use-exp1", action="store_true", help="Use Exp1 best_hcqt.pt instead of Exp3")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-umap", action="store_true", help="Skip UMAP (if umap-learn not installed)")
    args = parser.parse_args()
    seed = args.seed

    setup_logging(EXP4_RUN_LOG)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXP4_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Experiment 4 started (interpretability + FFT-MLP baseline)")

    for pkg in ["torch", "numpy", "sklearn", "matplotlib", "librosa", "pandas"]:
        try:
            __import__(pkg)
            log.info("  %s: OK", pkg)
        except ImportError as e:
            log.warning("  %s: MISSING — %s", pkg, e)
    try:
        __import__("umap")
        log.info("  umap-learn: OK")
    except ImportError:
        log.info("  umap-learn: not installed (use --no-umap to skip UMAP)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # HCQT model
    checkpoint = BEST_HCQT_EXP1 if args.use_exp1 else BEST_HCQT_EXP3
    if not checkpoint.exists():
        log.warning("Checkpoint not found: %s; using Exp1/Exp3 path as-is.", checkpoint)
    model = build_hcqt_cnn()
    load_hcqt_checkpoint(model, checkpoint)
    model = model.to(device)
    model.eval()

    # Synthetic HCQT splits (for Grad-CAM, t-SNE, eval)
    synthetic_splits = get_synthetic_hcqt_splits(seed=seed)
    synth_test_loader = get_synthetic_hcqt_test_loader(seed=seed, batch_size=BATCH_SIZE)
    recorded_train_loader, recorded_val_loader, recorded_test_loader = get_recorded_hcqt_loaders(
        seed=seed, batch_size=BATCH_SIZE
    )

    # Grad-CAM (synthetic and recorded test)
    run_gradcam(model, device, synthetic_splits, EXP4_GRADCAM_DIR, n_per_class=GRADCAM_N_SAMPLES_PER_CLASS)
    run_gradcam_recorded(model, device, recorded_test_loader, EXP4_GRADCAM_DIR, n_per_class=GRADCAM_N_SAMPLES_PER_CLASS)

    # Embeddings HCQT: synthetic test + recorded test
    emb_synth, pred_synth, y_synth = get_embeddings_and_predictions(model, synth_test_loader, device)
    emb_rec, pred_rec, y_rec = get_embeddings_and_predictions(model, recorded_test_loader, device)
    all_emb = np.concatenate([emb_synth, emb_rec])
    all_labels = np.concatenate([y_synth, y_rec])
    source_label = np.array([0] * len(y_synth) + [1] * len(y_rec))

    # t-SNE
    coords_tsne = run_tsne(all_emb, seed=seed)
    plot_embedding_2d(
        coords_tsne, all_labels,
        EXP4_FIGURES_DIR / "tsne_embeddings.png",
        title="HCQT embeddings (t-SNE); color=interval, marker=source",
        marker_by=source_label,
        marker_labels=["synthetic", "recorded"],
    )
    # UMAP
    if not args.no_umap:
        try:
            coords_umap = run_umap(all_emb)
            plot_embedding_2d(
                coords_umap, all_labels,
                EXP4_FIGURES_DIR / "umap_embeddings.png",
                title="HCQT embeddings (UMAP); color=interval, marker=source",
                marker_by=source_label,
                marker_labels=["synthetic", "recorded"],
            )
        except Exception as e:
            log.warning("UMAP failed: %s", e)

    # Confusion matrix HCQT
    acc_synth = float(accuracy_score(y_synth, pred_synth))
    acc_rec = float(accuracy_score(y_rec, pred_rec))
    plot_confusion_matrix(
        y_synth, pred_synth,
        EXP4_FIGURES_DIR / "confusion_hcqt_synthetic_test.png",
        title=f"HCQT CNN — Synthetic test (acc={acc_synth:.4f})",
        normalize="true",
    )
    plot_confusion_matrix(
        y_rec, pred_rec,
        EXP4_FIGURES_DIR / "confusion_hcqt_recorded_test.png",
        title=f"HCQT CNN — Recorded test (acc={acc_rec:.4f})",
        normalize="true",
    )

    # FFT-MLP (3 training regimes)
    fft_dim = get_fft_feature_dim()
    (train_s, val_s, test_s) = extract_fft_splits_synthetic(
        SYNTHETIC_METADATA, SYNTHETIC_DIR, seed, train_dev_fraction=TRAIN_DEV_FRACTION
    )
    (train_r, val_r, test_r) = extract_fft_splits_recorded(
        RECORDED_METADATA, RECORDED_DIR, RECORDED_VAL_FRACTION, RECORDED_SPLIT_BY_FILE, seed
    )

    fft_results = {}
    for regime, name in [
        ("synthetic_only", "Synthetic only"),
        ("recorded_only", "Recorded only"),
        ("combined", "Synthetic + Recorded"),
    ]:
        if regime == "synthetic_only":
            X_tr, y_tr = train_s[0], train_s[1]
            X_v, y_v = val_s[0], val_s[1]
            X_te, y_te = test_s[0], test_s[1]
            scaler = StandardScaler()
            scaler.fit(X_tr)
            X_tr = scaler.transform(X_tr).astype(np.float32)
            X_v = scaler.transform(X_v).astype(np.float32)
            X_te = scaler.transform(X_te).astype(np.float32)
            tl, vl, te = create_fft_dataloaders((X_tr, y_tr), (X_v, y_v), (X_te, y_te))
            test_eval = "synthetic"
        elif regime == "recorded_only":
            X_tr, y_tr = train_r[0], train_r[1]
            X_v, y_v = val_r[0], val_r[1]
            X_te, y_te = test_r[0], test_r[1]
            scaler = StandardScaler()
            scaler.fit(X_tr)
            X_tr = scaler.transform(X_tr).astype(np.float32)
            X_v = scaler.transform(X_v).astype(np.float32)
            X_te = scaler.transform(X_te).astype(np.float32)
            tl, vl, te = create_fft_dataloaders((X_tr, y_tr), (X_v, y_v), (X_te, y_te))
            test_eval = "recorded"
        else:
            X_tr = np.concatenate([train_s[0], train_r[0]])
            y_tr = np.concatenate([train_s[1], train_r[1]])
            X_v = np.concatenate([val_s[0], val_r[0]])
            y_v = np.concatenate([val_s[1], val_r[1]])
            scaler = StandardScaler()
            scaler.fit(X_tr)
            X_tr = scaler.transform(X_tr).astype(np.float32)
            X_v = scaler.transform(X_v).astype(np.float32)
            X_te_s = scaler.transform(test_s[0]).astype(np.float32)
            X_te_r = scaler.transform(test_r[0]).astype(np.float32)
            tl, vl, _ = create_fft_dataloaders((X_tr, y_tr), (X_v, y_v), (X_te_s, test_s[1]))
            te_syn = create_fft_dataloaders((X_tr[:1], y_tr[:1]), (X_v[:1], y_v[:1]), (X_te_s, test_s[1]))[2]
            te_rec = create_fft_dataloaders((X_tr[:1], y_tr[:1]), (X_v[:1], y_v[:1]), (X_te_r, test_r[1]))[2]
            test_eval = "both"

        mlp = build_fft_mlp(input_dim=fft_dim)
        mlp, _ = train_fft_mlp(mlp, tl, vl, device, max_epochs=MAX_EPOCHS, patience=PATIENCE)
        if test_eval == "both":
            res_syn = evaluate_fft_mlp(mlp, te_syn, device)
            res_rec = evaluate_fft_mlp(mlp, te_rec, device)
            fft_results[regime] = {
                "synthetic_test": res_syn,
                "recorded_test": res_rec,
                "name": name,
            }
            plot_confusion_matrix(
                res_syn["targets"], res_syn["preds"],
                EXP4_FIGURES_DIR / f"confusion_fft_mlp_{regime}_synthetic.png",
                title=f"FFT-MLP {name} — Synthetic test (acc={res_syn['accuracy']:.4f})",
                normalize="true",
            )
            plot_confusion_matrix(
                res_rec["targets"], res_rec["preds"],
                EXP4_FIGURES_DIR / f"confusion_fft_mlp_{regime}_recorded.png",
                title=f"FFT-MLP {name} — Recorded test (acc={res_rec['accuracy']:.4f})",
                normalize="true",
            )
            # FFT-MLP embeddings t-SNE/UMAP (synthetic + recorded test)
            emb_s, _, y_s = get_fft_embeddings_and_predictions(mlp, te_syn, device)
            emb_r, _, y_r = get_fft_embeddings_and_predictions(mlp, te_rec, device)
            all_emb = np.concatenate([emb_s, emb_r])
            all_y = np.concatenate([y_s, y_r])
            src = np.array([0] * len(y_s) + [1] * len(y_r))
            coords_tsne = run_tsne(all_emb, seed=seed)
            plot_embedding_2d(
                coords_tsne, all_y,
                EXP4_FIGURES_DIR / f"tsne_fft_mlp_{regime}.png",
                title=f"FFT-MLP {name} embeddings (t-SNE)",
                marker_by=src, marker_labels=["synthetic", "recorded"],
            )
            if not args.no_umap:
                try:
                    coords_umap = run_umap(all_emb)
                    plot_embedding_2d(
                        coords_umap, all_y,
                        EXP4_FIGURES_DIR / f"umap_fft_mlp_{regime}.png",
                        title=f"FFT-MLP {name} embeddings (UMAP)",
                        marker_by=src, marker_labels=["synthetic", "recorded"],
                    )
                except Exception as e:
                    log.warning("UMAP FFT %s failed: %s", regime, e)
        else:
            res = evaluate_fft_mlp(mlp, te, device)
            fft_results[regime] = {"test": res, "name": name}
            plot_confusion_matrix(
                res["targets"], res["preds"],
                EXP4_FIGURES_DIR / f"confusion_fft_mlp_{regime}.png",
                title=f"FFT-MLP {name} (acc={res['accuracy']:.4f})",
                normalize="true",
            )
            # FFT-MLP embeddings t-SNE/UMAP (single test set)
            emb, _, y = get_fft_embeddings_and_predictions(mlp, te, device)
            coords_tsne = run_tsne(emb, seed=seed)
            plot_embedding_2d(
                coords_tsne, y,
                EXP4_FIGURES_DIR / f"tsne_fft_mlp_{regime}.png",
                title=f"FFT-MLP {name} embeddings (t-SNE)",
            )
            if not args.no_umap:
                try:
                    coords_umap = run_umap(emb)
                    plot_embedding_2d(
                        coords_umap, y,
                        EXP4_FIGURES_DIR / f"umap_fft_mlp_{regime}.png",
                        title=f"FFT-MLP {name} embeddings (UMAP)",
                    )
                except Exception as e:
                    log.warning("UMAP FFT %s failed: %s", regime, e)

    report_lines = [
        "=" * 60,
        "Experiment 4 — Interpretability and FFT baseline",
        "=" * 60,
        "",
        "HCQT CNN (best checkpoint: %s)" % str(checkpoint),
        "  Synthetic test accuracy: %.4f" % acc_synth,
        "  Recorded test accuracy:  %.4f" % acc_rec,
        "",
        "Grad-CAM (synthetic + recorded): heatmaps in %s" % str(EXP4_GRADCAM_DIR),
        "t-SNE/UMAP HCQT and FFT-MLP: figures in %s" % str(EXP4_FIGURES_DIR),
        "",
        "FFT-MLP baseline:",
    ]
    for regime, data in fft_results.items():
        report_lines.append("  %s" % data["name"])
        if "synthetic_test" in data:
            report_lines.append("    Synthetic test acc: %.4f  macro_f1: %.4f" % (
                data["synthetic_test"]["accuracy"], data["synthetic_test"]["macro_f1"]))
            report_lines.append("    Recorded test acc:  %.4f  macro_f1: %.4f" % (
                data["recorded_test"]["accuracy"], data["recorded_test"]["macro_f1"]))
        else:
            report_lines.append("    Test acc: %.4f  macro_f1: %.4f" % (
                data["test"]["accuracy"], data["test"]["macro_f1"]))
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    EXP4_RESULTS_TXT.parent.mkdir(parents=True, exist_ok=True)
    EXP4_RESULTS_TXT.write_text(report_text, encoding="utf-8")
    log.info("Report written to %s", EXP4_RESULTS_TXT)
    log.info("Experiment 4 finished.")


if __name__ == "__main__":
    main()
