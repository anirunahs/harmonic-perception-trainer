"""
Experiment 3 — Fine-tuning on recorded (microphone) dataset.

Entry point: python -m experiments.exp3_microphone [OPTIONS]

Loads best HCQT CNN (Exp 1 architecture), optionally from checkpoint.
Fine-tunes on NF (lid) segments; early stopping on val (NF holdout);
evaluates on MF (table).
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

from .config import (
    BATCH_SIZE,
    CACHE_DIR,
    CHECKPOINT_DIR,
    CLASS_SPLIT_TEST,
    CLASS_SPLIT_TRAIN,
    CLASS_SPLIT_VAL,
    EXP3_BEST_CHECKPOINT,
    EXP3_LOG,
    EXP3_RESULTS_TXT,
    EXP3_RUN_LOG,
    MAX_EPOCHS,
    PATIENCE,
    PRETRAINED_CQT,
    PRETRAINED_HCQT,
    PRETRAINED_MEL,
    RECORDED_METADATA_CSV,
    RECORDED_SEGMENTS_DIR,
    RESULTS_DIR,
    SEEDS,
    SYNTHETIC_DATASET_DIR,
    SYNTHETIC_METADATA_CSV,
    SYNTHETIC_TEST_INSTRUMENTS,
    T_MAX,
    TEST_POSITIONS,
    TRAIN_POSITIONS,
)
from .dataset import (
    create_dataloaders,
    create_synthetic_test_loader,
    prepare_recorded_splits,
    prepare_recorded_splits_by_class,
)
from .evaluator import compute_metrics, format_results, format_results_comparison
from .models import build_model, load_pretrained
from .trainer import TrainHistory, train, evaluate_loader
from torch.nn import CrossEntropyLoss

log = logging.getLogger("exp3")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from .config import (
        RECORDED_VAL_FRACTION,
        RECORDED_SPLIT_BY_FILE,
        TRAIN_POSITIONS,
        TEST_POSITIONS,
    )
    p = argparse.ArgumentParser(
        description="Experiment 3: fine-tune HCQT on recorded (NF/MF) dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    p.add_argument("--metadata", type=Path, default=RECORDED_METADATA_CSV, help="Recorded metadata CSV")
    p.add_argument("--segments-dir", type=Path, default=RECORDED_SEGMENTS_DIR, help="Processed segments root")
    p.add_argument("--pretrained", type=Path, default=None, help="Path to Exp 1 checkpoint (optional, for single --representation)")
    p.add_argument("--no-pretrained", action="store_true", help="Do not load pretrained; train from scratch on recorded")
    p.add_argument(
        "--representation",
        choices=["hcqt", "mel", "cqt", "hcqt_scratch", "all"],
        default="all",
        help="Which model to run: hcqt/mel/cqt (pretrained from Exp 1), hcqt_scratch (recorded only), or all (compare all four).",
    )
    p.add_argument("--val-fraction", type=float, default=RECORDED_VAL_FRACTION)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--no-cache", action="store_true", help="Disable feature cache")
    p.add_argument("--output", type=Path, default=EXP3_RESULTS_TXT)
    p.add_argument(
        "--split",
        choices=["position", "class"],
        default="position",
        help="position: train/val=NF, test=MF (robustness to mic). "
             "class: train=6 classes NF, val=2 NF, test=4 held-out classes MF (no content leakage).",
    )
    p.add_argument(
        "--eval-synthetic",
        action="store_true",
        help="After fine-tuning, also evaluate on synthetic test set (violin, flute).",
    )
    p.add_argument(
        "--zero-shot",
        action="store_true",
        help="Also evaluate Exp 1 HCQT on MF test without any fine-tuning (baseline).",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    import sys
    level = logging.DEBUG if args.verbose else logging.INFO
    log_fmt = "%(asctime)s  %(name)-6s  %(levelname)-7s  %(message)s"
    date_fmt = "%H:%M:%S"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(EXP3_LOG, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(log_fmt, datefmt=date_fmt))
    logging.basicConfig(
        level=level,
        format=log_fmt,
        datefmt=date_fmt,
        stream=sys.stdout,
        force=True,
    )

    logging.getLogger().addHandler(file_handler)
    log.info("Log file: %s", EXP3_LOG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s - training on GPU", torch.cuda.get_device_name(0))
    else:
        log.info("Training on CPU")

    if not args.metadata.exists():
        log.error("Recorded metadata not found at %s. Run the recorded dataset pipeline first.", args.metadata)
        sys.exit(1)

    from .config import RECORDED_SPLIT_BY_FILE

    cache_dir = None if args.no_cache else CACHE_DIR
    seeds = SEEDS[: args.seeds]
    split_sizes: dict = {}

    if args.representation == "all":
        conditions = [
            ("hcqt", PRETRAINED_HCQT),
            ("mel", PRETRAINED_MEL),
            ("cqt", PRETRAINED_CQT),
            ("hcqt_scratch", None),
        ]
    else:
        single = args.representation
        ckpt_map = {"hcqt": PRETRAINED_HCQT, "mel": PRETRAINED_MEL, "cqt": PRETRAINED_CQT, "hcqt_scratch": None}
        conditions = [(single, args.pretrained if args.pretrained else (None if args.no_pretrained else ckpt_map.get(single)))]

    results_by_repr: dict[str, list] = {}
    histories_by_repr: dict[str, list[TrainHistory]] = {}
    best_mean_acc = -1.0
    best_state = None
    best_repr_name = None
    criterion = CrossEntropyLoss()

    # Zero-shot: Exp 1 HCQT on MF test without fine-tuning
    if args.zero_shot and Path(PRETRAINED_HCQT).exists():
        log.info("Evaluating HCQT (Exp 1) on MF test without fine-tuning (zero-shot)")
        zero_metrics_list: list[dict] = []
        for seed_idx, seed in enumerate(seeds):
            set_seed(seed)
            if args.split == "position":
                df_train, df_val, df_test = prepare_recorded_splits(
                    args.metadata, args.segments_dir,
                    TRAIN_POSITIONS, TEST_POSITIONS,
                    args.val_fraction, RECORDED_SPLIT_BY_FILE, seed,
                )
            else:
                df_train, df_val, df_test = prepare_recorded_splits_by_class(
                    args.metadata, TRAIN_POSITIONS, TEST_POSITIONS,
                    CLASS_SPLIT_TRAIN, CLASS_SPLIT_VAL, CLASS_SPLIT_TEST,
                )
            if seed_idx == 0:
                split_sizes = {"train": len(df_train), "val": len(df_val), "test": len(df_test)}
            _, _, test_loader = create_dataloaders(
                df_train, df_val, df_test,
                args.segments_dir, cache_dir, args.batch_size,
                representation="hcqt", use_augment=False,
            )
            model = build_model("hcqt")
            load_pretrained(model, Path(PRETRAINED_HCQT), strict=True)
            model = model.to(device)
            test_metrics = evaluate_loader(model, test_loader, criterion, device)
            m = compute_metrics(test_metrics["preds"], test_metrics["targets"])
            zero_metrics_list.append(m)
            log.info("HCQT zero-shot (seed %d) test MF acc %.4f  f1 %.4f", seed, m["accuracy"], m["macro_f1"])
        results_by_repr["hcqt_zero_shot"] = zero_metrics_list
        histories_by_repr["hcqt_zero_shot"] = []

    for repr_name, pretrained_path in conditions:
        repr_for_data = "hcqt" if repr_name == "hcqt_scratch" else repr_name
        all_metrics: list[dict] = []
        all_histories: list[TrainHistory] = []

        for seed_idx, seed in enumerate(seeds):
            set_seed(seed)
            log.info("-- %s Seed %d/%d (%d) --", repr_name, seed_idx + 1, len(seeds), seed)

            if args.split == "position":
                df_train, df_val, df_test = prepare_recorded_splits(
                    args.metadata,
                    args.segments_dir,
                    TRAIN_POSITIONS,
                    TEST_POSITIONS,
                    args.val_fraction,
                    RECORDED_SPLIT_BY_FILE,
                    seed,
                )
            else:
                df_train, df_val, df_test = prepare_recorded_splits_by_class(
                    args.metadata,
                    TRAIN_POSITIONS,
                    TEST_POSITIONS,
                    CLASS_SPLIT_TRAIN,
                    CLASS_SPLIT_VAL,
                    CLASS_SPLIT_TEST,
                )
            split_sizes = {"train": len(df_train), "val": len(df_val), "test": len(df_test)}
            if seed_idx == 0:
                log.info("Split mode: %s - train: %d  val: %d  test: %d", args.split, split_sizes["train"], split_sizes["val"], split_sizes["test"])

            train_loader, val_loader, test_loader = create_dataloaders(
                df_train, df_val, df_test,
                args.segments_dir,
                cache_dir,
                args.batch_size,
                representation=repr_for_data,
                use_augment=True,
            )

            model = build_model(repr_for_data)
            ckpt = Path(pretrained_path) if pretrained_path and Path(pretrained_path).exists() else None
            if ckpt:
                load_pretrained(model, ckpt, strict=True)
                log.info("Loaded pretrained from %s", ckpt)
            elif pretrained_path and not args.no_pretrained and args.representation != "all":
                log.warning("Pretrained checkpoint not found at %s; training from scratch", pretrained_path)

            history = train(
                model,
                train_loader,
                val_loader,
                device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                t_max=T_MAX,
            )
            all_histories.append(history)

            test_metrics = evaluate_loader(model, test_loader, criterion, device)
            metrics = compute_metrics(test_metrics["preds"], test_metrics["targets"])
            all_metrics.append(metrics)
            log.info("%s Seed %d done - test acc %.4f  f1 %.4f", repr_name, seed, metrics["accuracy"], metrics["macro_f1"])

            if args.eval_synthetic and SYNTHETIC_METADATA_CSV.exists() and repr_for_data == "hcqt" and seed_idx == 0:
                syn_loader = create_synthetic_test_loader(
                    SYNTHETIC_METADATA_CSV,
                    SYNTHETIC_DATASET_DIR,
                    cache_dir,
                    args.batch_size,
                    SYNTHETIC_TEST_INSTRUMENTS,
                )
                syn_metrics = evaluate_loader(model, syn_loader, criterion, device)
                syn_agg = compute_metrics(syn_metrics["preds"], syn_metrics["targets"])
                log.info("Synthetic test (violin, flute) acc %.4f  f1 %.4f", syn_agg["accuracy"], syn_agg["macro_f1"])

        results_by_repr[repr_name] = all_metrics
        histories_by_repr[repr_name] = all_histories
        mean_acc = float(np.mean([m["accuracy"] for m in all_metrics]))
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_state = copy.deepcopy(model.state_dict())
            best_repr_name = repr_name

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save({"model_state_dict": best_state, "representation": best_repr_name}, EXP3_BEST_CHECKPOINT)
        log.info("Saved best Exp 3 model (%s, test acc %.4f) to %s", best_repr_name, best_mean_acc, EXP3_BEST_CHECKPOINT)

    if len(conditions) > 1:
        report = format_results_comparison(
            results_by_repr, histories_by_repr, split_sizes, args.output, split_mode=args.split,
        )
    else:
        repr_name = conditions[0][0]
        report = format_results(
            results_by_repr[repr_name],
            split_sizes,
            histories_by_repr[repr_name],
            args.output,
            split_mode=args.split,
            synthetic_metrics=None,
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXP3_RUN_LOG.write_text(
        f"Experiment 3 completed. Representation(s): {[c[0] for c in conditions]}. Seeds: {seeds}. Output: {args.output}\n",
        encoding="utf-8",
    )
    log.info("Final results written to %s", args.output)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
