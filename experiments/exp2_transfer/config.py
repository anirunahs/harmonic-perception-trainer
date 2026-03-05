"""
Experiment 2 — Representation and Transfer for Interval Recognition.

RQ: (A) How do from-scratch models with different representations (mel, CQT, HCQT)
    compare on the same dataset? (B) How effective is transfer from
    general-purpose audio models (PANNs, AST) vs. task-aligned representation?

Design: Same dataset and splits as Experiment 1. We compare (1) from-scratch
    baselines with mel, CQT, and HCQT (same architectures as Exp 1); (2) transfer
    from AudioSet (PANNs CNN14, AST) at three fine-tuning depths. This makes
    the experiment "interesting": representation-from-scratch vs. pretrained
    general audio in one place.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset-preparation" / "synthetic-dataset"
METADATA_CSV = DATASET_DIR / "metadata.csv"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
RESULTS_DIR = PROJECT_ROOT / "ztu" / "docs" / "experiments"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"

# Audio
SR = 22050
DURATION = 2.0
NUM_SAMPLES = int(SR * DURATION)

# Labels
INTERVAL_CLASSES = [
    "minor_2nd", "major_2nd", "minor_3rd", "major_3rd",
    "perfect_4th", "tritone", "perfect_5th", "minor_6th",
    "major_6th", "minor_7th", "major_7th", "perfect_8th",
]
NUM_CLASSES = len(INTERVAL_CLASSES)
LABEL_TO_IDX = {name: i for i, name in enumerate(INTERVAL_CLASSES)}
IDX_TO_LABEL = {i: name for i, name in enumerate(INTERVAL_CLASSES)}

# Instrument-based split
SPLIT_INSTRUMENTS = {
    "train": [
        "piano", "guitar_nylon", "guitar_steel",
        "cello", "church_organ", "trumpet",
    ],
    "val": ["clarinet", "marimba"],
    "test": ["violin", "flute"],
}
TRAIN_DEV_FRACTION = 0.15

# PANNs CNN14
PANNS_CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/"
    "Cnn14_mAP%3D0.431.pth"
)
PANNS_CHECKPOINT_FNAME = "Cnn14_mAP=0.431.pth"
PANNS_SR = 32000
PANNS_N_FFT = 1024
PANNS_HOP = 320
PANNS_N_MELS = 64
PANNS_FMIN = 50.0
PANNS_FMAX = 14000.0
PANNS_EMBED_DIM = 2048

# AST
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR = 16000
AST_TARGET_LENGTH = 204          # ~2 s at 10 ms hop; pos-embeddings interpolated
AST_N_MELS = 128
AST_FBANK_MEAN = -4.2677393     # AudioSet training-set statistics
AST_FBANK_STD = 4.5689974
AST_EMBED_DIM = 768

# From-scratch representations
BASELINE_N_MELS = 128
BASELINE_N_FFT = 2048
BASELINE_HOP = 512
BASELINE_FMIN = 80.0
BASELINE_FMAX = 8000.0
# CQT / HCQT
CQT_HOP = 512
CQT_BINS_PER_OCTAVE = 36
CQT_N_BINS = 252
CQT_FMIN = 65.41  # C2
HCQT_HARMONICS = (1, 2, 3, 4, 5)

# Fine-tuning level hyperparameters
LEVEL_HPARAMS = {
    "frozen": dict(
        backbone_lr=0.0,
        head_lr=1e-3,
        weight_decay=1e-4,
        max_epochs=40,
        patience=10,
        batch_size_panns=64,
        batch_size_ast=32,
    ),
    "partial": dict(
        backbone_lr=1e-4,
        head_lr=1e-3,
        weight_decay=1e-4,
        max_epochs=100,
        patience=15,
        batch_size_panns=32,
        batch_size_ast=16,
    ),
    "full": dict(
        backbone_lr=1e-5,
        head_lr=1e-3,
        weight_decay=1e-4,
        max_epochs=80,
        patience=15,
        batch_size_panns=16,
        batch_size_ast=8,
    ),
}

SCRATCH_HPARAMS = dict(
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=150,
    patience=20,
    batch_size=32,
)

# Shared training
DROPOUT = 0.3
CLASSIFIER_HIDDEN = 256
T_MAX_TRANSFER = 30
T_MAX_SCRATCH = 50
MIN_DELTA = 1e-4
USE_AMP = True

SEEDS = [42, 43, 44]

PANNS_UNFREEZE_BLOCKS = 2       # last N conv blocks for partial
AST_UNFREEZE_LAYERS = 2         # last N transformer layers for partial

# SpecAugment (scratch baseline only)
FREQ_MASK_PARAM = 15
TIME_MASK_PARAM = 10
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

CONDITIONS = [
    ("cnn_mel_scratch",  "scratch_mel",  None),
    ("cnn_cqt_scratch",  "scratch_cqt",  None),
    ("cnn_hcqt_scratch", "scratch_hcqt", None),
    ("ast_frozen",       "ast",          "frozen"),
    ("ast_partial",      "ast",          "partial"),
    ("ast_full",         "ast",          "full"),
    ("panns_frozen",     "panns",       "frozen"),
    ("panns_partial",    "panns",       "partial"),
    ("panns_full",      "panns",       "full"),
]
