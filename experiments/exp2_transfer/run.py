"""
Experiment 2 — Transfer Learning from Pretrained Audio Models.

Entry point::

    python -m experiments.exp2_transfer [OPTIONS]

Compares PANNs CNN14 and AST at three fine-tuning depths (frozen,
partial, full) plus a from-scratch mel-CNN baseline, all on the same
synthetic interval-recognition dataset and instrument-based splits as
Experiment 1.  Writes a comprehensive results file.
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import (
    CONDITIONS,
    DROPOUT,
    LEVEL_HPARAMS,
    METADATA_CSV,
    RESULTS_DIR,
    SCRATCH_HPARAMS,
    SEEDS,
    T_MAX_SCRATCH,
    T_MAX_TRANSFER,
)
from .dataset import create_dataloaders, prepare_splits
from .evaluator import (
    aggregate_seeds,
    evaluate_all_splits,
    format_results,
    load_and_merge_condition_results,
    save_condition_results,
)
from .models import build_scratch_model, build_transfer_model
from .trainer import TrainHistory, TransferTrainer

log = logging.getLogger("exp2")


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
        description="Experiment 2: transfer learning comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--conditions", nargs="+",
        default=[c[0] for c in CONDITIONS],
        help="Which conditions to evaluate",
    )
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of random seeds (1–5)")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable feature caching")
    p.add_argument("--output", type=str,
                   default=str(RESULTS_DIR / "exp2_results.txt"))
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _get_condition(name: str):
    for c in CONDITIONS:
        if c[0] == name:
            return c
    raise ValueError(f"Unknown condition: {name}")


# Main

def run(args: argparse.Namespace) -> None:
    log_level = logging.DEBUG if args.verbose else logging.INFO
    fmt = "%(asctime)s  %(name)-6s  %(levelname)-7s  %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(level=log_level, format=fmt, datefmt=datefmt)

    log_file = Path(args.output).with_suffix(".run.log")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s (%.1f GB VRAM) — training on GPU", props.name,
                 props.total_memory / 1e9)
    else:
        log.warning("CUDA not available — training on CPU (will be slow)")

    if not METADATA_CSV.exists():
        log.error("Metadata not found: %s", METADATA_CSV)
        sys.exit(1)

    seeds = SEEDS[:args.seeds]

    all_cond_results: Dict[str, Dict] = {}
    all_histories: Dict[str, List[TrainHistory]] = {}
    condition_info: Dict[str, Dict] = {}
    split_sizes: Dict[str, int] = {}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp1_ref = {
        "HCQT CNN (Exp 1)": "acc 0.9983 ± 0.0023,  f1 0.9981 ± 0.0024",
        "CQT CNN  (Exp 1)": "acc 0.9943 ± 0.0045,  f1 0.9935 ± 0.0051",
        "Mel CNN  (Exp 1)": "acc 0.9537 ± 0.0264,  f1 0.9521 ± 0.0274",
    }

    for cond_name in args.conditions:
        name, model_type, ft_level = _get_condition(cond_name)
        log.info("\n%s\n  Condition: %s  (model=%s, level=%s)\n%s",
                 "=" * 64, name, model_type, ft_level, "=" * 64)

        try:
            splits = prepare_splits(
                model_type, seed=seeds[0], use_cache=not args.no_cache)

            if not split_sizes:
                split_sizes = {k: len(v[1]) for k, v in splits.items()}

            is_scratch = model_type in ("scratch_mel", "scratch_cqt", "scratch_hcqt")
            if is_scratch:
                hp = SCRATCH_HPARAMS.copy()
                batch_size = hp["batch_size"]
                t_max = T_MAX_SCRATCH
            else:
                hp = LEVEL_HPARAMS[ft_level].copy()
                batch_size = hp.get(f"batch_size_{model_type}",
                                    hp.get("batch_size_panns", 32))
                t_max = T_MAX_TRANSFER

            info: Dict = {"model": model_type,
                           "finetune_level": ft_level or "from_scratch",
                           "batch_size": batch_size}
            if is_scratch:
                info.update(lr=hp["lr"], weight_decay=hp["weight_decay"],
                            max_epochs=hp["max_epochs"],
                            patience=hp["patience"],
                            optimizer="AdamW",
                            scheduler=f"CosineAnnealingLR(T_max={t_max})",
                            loss="CrossEntropyLoss",
                            spec_augment=True)
            else:
                info.update(backbone_lr=hp["backbone_lr"],
                            head_lr=hp["head_lr"],
                            weight_decay=hp["weight_decay"],
                            max_epochs=hp["max_epochs"],
                            patience=hp["patience"],
                            optimizer="AdamW",
                            scheduler=f"CosineAnnealingLR(T_max={t_max})",
                            loss="CrossEntropyLoss",
                            amp=True)
            condition_info[name] = info

            seed_metrics: List[Dict] = []
            seed_histories: List[TrainHistory] = []

            for i, seed in enumerate(seeds):
                log.info("── Seed %d/%d (%d) ──", i + 1, len(seeds), seed)
                set_seed(seed)

                loaders = create_dataloaders(splits, batch_size, model_type)

                if is_scratch:
                    repr_name = "mel" if model_type == "scratch_mel" else (
                        "cqt" if model_type == "scratch_cqt" else "hcqt")
                    model = build_scratch_model(
                        representation=repr_name, dropout=DROPOUT).to(device)
                    trainer = TransferTrainer(
                        model, loaders, device,
                        backbone_lr=hp["lr"], head_lr=hp["lr"],
                        weight_decay=hp["weight_decay"],
                        max_epochs=hp["max_epochs"],
                        patience=hp["patience"],
                        t_max=t_max,
                    )
                else:
                    model = build_transfer_model(
                        model_type, ft_level, device, dropout=DROPOUT)
                    trainer = TransferTrainer(
                        model, loaders, device,
                        backbone_lr=hp["backbone_lr"],
                        head_lr=hp["head_lr"],
                        weight_decay=hp["weight_decay"],
                        max_epochs=hp["max_epochs"],
                        patience=hp["patience"],
                        t_max=t_max,
                    )

                history = trainer.fit()
                seed_histories.append(history)

                metrics = evaluate_all_splits(model, loaders, device)
                seed_metrics.append(metrics)
                log.info("  Seed %d done — test acc %.3f  f1 %.3f",
                         seed, metrics["test"]["accuracy"],
                         metrics["test"]["macro_f1"])

                del model, trainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_cond_results[name] = aggregate_seeds(seed_metrics)
            all_histories[name] = seed_histories

            del splits, loaders
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            report = format_results(
                all_cond_results, all_histories, condition_info,
                split_sizes, exp1_reference=exp1_ref,
            )
            output_path.write_text(report, encoding="utf-8")
            log.info("Intermediate results saved to %s", output_path)
            if len(args.conditions) == 1:
                save_condition_results(
                    name, all_cond_results[name], condition_info[name],
                    split_sizes, seed_histories,
                    RESULTS_DIR / f"exp2_{name}_results.json",
                )

        except Exception:
            log.exception("Condition %s FAILED — skipping", name)
            continue

    # Console summary
    print("\n" + "=" * 64)
    print("  EXPERIMENT 2 — SUMMARY (test set)")
    print("=" * 64)
    for cn, agg in all_cond_results.items():
        t = agg["test"]
        print(f"  {cn:<20s}  acc {t['accuracy']['mean']:.3f}±"
              f"{t['accuracy']['std']:.3f}  "
              f"f1 {t['macro_f1']['mean']:.3f}±"
              f"{t['macro_f1']['std']:.3f}")
    print()
    log.info("Final results written to %s", output_path)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
