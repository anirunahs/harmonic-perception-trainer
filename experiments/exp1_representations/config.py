"""
Experiment 1 — Time-Frequency Representation Comparison.

RQ: Which input representation (Mel / CQT / HCQT / raw waveform) is best
    suited for musical interval recognition, and is an explicit
    time-frequency transform necessary at all?

All constants, hyper-parameters, and split definitions live here so that
every other module in the experiment imports a single source of truth.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset-preparation" / "synthetic-dataset"
METADATA_CSV = DATASET_DIR / "metadata.csv"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"
BEST_HCQT_CHECKPOINT = CHECKPOINT_DIR / "best_hcqt.pt"
BEST_MEL_CHECKPOINT = CHECKPOINT_DIR / "best_mel.pt"
BEST_CQT_CHECKPOINT = CHECKPOINT_DIR / "best_cqt.pt"
RESULTS_DIR = PROJECT_ROOT / "ztu" / "docs" / "experiments"

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

# Fraction of training data reserved for train-dev (bias / variance diagnosis)
TRAIN_DEV_FRACTION = 0.15

# Representation parameters
MEL_PARAMS = dict(
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=80.0,
    fmax=8000.0,
)

CQT_PARAMS = dict(
    hop_length=512,
    bins_per_octave=36,
    n_bins=252,
    fmin=65.41,           # C2
)

HCQT_PARAMS = dict(
    hop_length=512,
    bins_per_octave=36,
    n_bins=252,
    fmin=65.41,
    harmonics=(1, 2, 3, 4, 5),
)

RAW_PARAMS = dict(
    num_samples=NUM_SAMPLES,
)

# Default: Mel, CQT, HCQT only. Raw (1D CNN) excluded — poor cross-timbral
# generalization (~40% test) and long training; see ztu/docs/experiments/exp1_analysis.txt
REPRESENTATIONS = {
    "mel":  MEL_PARAMS,
    "cqt":  CQT_PARAMS,
    "hcqt": HCQT_PARAMS,
    "raw":  RAW_PARAMS,
}
DEFAULT_REPRESENTATIONS = ["mel", "cqt", "hcqt"]

# CNN architecture
CNN2D_CHANNELS = (32, 64, 128, 256)
CNN2D_KERNEL = 3
CLASSIFIER_DIM = 128

CNN1D_CHANNELS = (64, 128, 256, 256)
CNN1D_KERNELS = (256, 64, 32, 16)
CNN1D_FIRST_STRIDE = 64

# Training defaults
BATCH_SIZE = 32
MAX_EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
PATIENCE = 20          # early-stopping patience (epochs without improvement)
MIN_DELTA = 1e-4
T_MAX = 50             # CosineAnnealingLR period

SEEDS = [42, 43, 44, 45, 46]

# SpecAugment (applied to 2-D representations during training)
FREQ_MASK_PARAM = 15
TIME_MASK_PARAM = 10
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

# Optuna
OPTUNA_TRIALS = 50
OPTUNA_METRIC = "val_macro_f1"
OPTUNA_LR_RANGE = (1e-5, 1e-2)
OPTUNA_WD_RANGE = (1e-6, 1e-2)
OPTUNA_DROPOUT_RANGE = (0.1, 0.5)
OPTUNA_BATCH_SIZES = (16, 32, 64)
