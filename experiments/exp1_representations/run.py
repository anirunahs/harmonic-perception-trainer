"""
Experiment 1 — Time-Frequency Representation Comparison.

Entry point: ``python -m experiments.exp1_representations.run [OPTIONS]``

Compares Mel-spectrogram, CQT, HCQT, and raw-waveform 1-D CNN on the
same synthetic dataset and writes a comprehensive results file.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import (
    BATCH_SIZE, BEST_CQT_CHECKPOINT, BEST_HCQT_CHECKPOINT, BEST_MEL_CHECKPOINT,
    DEFAULT_REPRESENTATIONS, DROPOUT, LR, MAX_EPOCHS, PATIENCE, T_MAX, WEIGHT_DECAY,
    METADATA_CSV, REPRESENTATIONS, RESULTS_DIR, SEEDS,
)
from .dataset import create_dataloaders, prepare_splits
from .evaluator import (
    aggregate_seeds, evaluate_all_splits, format_results,
)
from .models import build_model
from .trainer import Trainer, TrainHistory, run_optuna

log = logging.getLogger("exp1")


# Reproducibility

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 1: representation comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--representations", nargs="+",
        default=DEFAULT_REPRESENTATIONS,
        choices=list(REPRESENTATIONS.keys()),
        help="Which representations to evaluate (default: mel, cqt, hcqt)",
    )
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of random seeds (1–5)")
    p.add_argument("--tune", action="store_true",
                   help="Run Optuna HPO before final training")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--no-cache", action="store_true",
                   help="Disable feature caching (re-extract every run)")
    p.add_argument("--output", type=str,
                   default=str(RESULTS_DIR / "exp1_results.txt"),
                   help="Path for the results text file")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


# Main logic

def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-6s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    if not METADATA_CSV.exists():
        log.error("Metadata not found at %s. Generate the synthetic "
                  "dataset first.", METADATA_CSV)
        sys.exit(1)

    seeds = SEEDS[:args.seeds]

    all_repr_results: Dict[str, Dict] = {}
    all_histories: Dict[str, List[TrainHistory]] = {}
    hyperparams: Dict[str, Dict] = {}
    split_sizes: Dict[str, int] = {}

    for repr_name in args.representations:
        log.info("\n%s\n  Representation: %s\n%s",
                 "=" * 60, repr_name.upper(), "=" * 60)

        # Features (cached across seeds)
        base_seed = seeds[0]
        splits = prepare_splits(repr_name, seed=base_seed,
                                use_cache=not args.no_cache)

        if not split_sizes:
            split_sizes = {k: len(v[1]) for k, v in splits.items()}

        # Optuna HPO
        if args.tune:
            best_hp = run_optuna(repr_name, splits, device)
        else:
            best_hp = {
                "lr": args.lr,
                "weight_decay": WEIGHT_DECAY,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
            }

        hp_record = {
            **best_hp,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "optimizer": "Adam",
            "scheduler": f"CosineAnnealingLR(T_max={T_MAX})",
            "loss": "CrossEntropyLoss",
            "spec_augment": repr_name != "raw",
        }
        hyperparams[repr_name] = hp_record

        # Multi-seed training
        seed_metrics: List[Dict] = []
        seed_histories: List[TrainHistory] = []

        for i, seed in enumerate(seeds):
            log.info("── Seed %d/%d (%d) ──", i + 1, len(seeds), seed)
            set_seed(seed)

            loaders = create_dataloaders(
                splits,
                batch_size=best_hp.get("batch_size", args.batch_size),
                repr_name=repr_name,
            )
            model = build_model(repr_name,
                                dropout=best_hp.get("dropout", args.dropout))

            trainer = Trainer(
                model, loaders, device,
                lr=best_hp.get("lr", args.lr),
                weight_decay=best_hp.get("weight_decay", WEIGHT_DECAY),
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            history = trainer.fit()
            seed_histories.append(history)

            metrics = evaluate_all_splits(model, loaders, device)
            seed_metrics.append(metrics)

        all_repr_results[repr_name] = aggregate_seeds(seed_metrics)
        all_histories[repr_name] = seed_histories

        # Save checkpoints
        if repr_name in ("hcqt", "mel", "cqt"):
            path = { "hcqt": BEST_HCQT_CHECKPOINT, "mel": BEST_MEL_CHECKPOINT, "cqt": BEST_CQT_CHECKPOINT }[repr_name]
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, path)
            log.info("Saved %s checkpoint to %s", repr_name.upper(), path)

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = format_results(
        all_repr_results, all_histories, hyperparams, split_sizes,
    )
    output_path.write_text(report, encoding="utf-8")
    log.info("Results written to %s", output_path)

    # Print summary to console
    print("\n" + "=" * 60)
    print("  SUMMARY (test set)")
    print("=" * 60)
    for rn, agg in all_repr_results.items():
        t = agg["test"]
        print(f"  {rn:<8s}  acc {t['accuracy']['mean']:.3f}±"
              f"{t['accuracy']['std']:.3f}  "
              f"f1 {t['macro_f1']['mean']:.3f}±{t['macro_f1']['std']:.3f}")
    print()



def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
