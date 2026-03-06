"""
Experiment 3 — Fine-tuning on recorded (microphone) dataset.

RQ: Does a model trained on synthetic data adapt to real microphone
    recordings (NF = lid + cleaned; MF = table) and retain accuracy?

Design: Load best HCQT CNN (Exp 1 architecture). Fine-tune only on
    train_recorded (NF); early stopping on val_recorded (holdout from NF);
    evaluate on test_recorded (MF).
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDED_METADATA_CSV = PROJECT_ROOT / "dataset-preparation" / "processed-segments" / "metadata.csv"
RECORDED_SEGMENTS_DIR = PROJECT_ROOT / "dataset-preparation" / "processed-segments"
EXP1_CHECKPOINTS = PROJECT_ROOT / "experiments" / "exp1_representations" / ".cache" / "checkpoints"
PRETRAINED_HCQT = EXP1_CHECKPOINTS / "best_hcqt.pt"
PRETRAINED_MEL = EXP1_CHECKPOINTS / "best_mel.pt"
PRETRAINED_CQT = EXP1_CHECKPOINTS / "best_cqt.pt"
# Backward compat
PRETRAINED_CHECKPOINT = PRETRAINED_HCQT
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
RESULTS_DIR = PROJECT_ROOT / "ztu" / "docs" / "experiments"
LOG_DIR = PROJECT_ROOT
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"
EXP3_BEST_CHECKPOINT = CHECKPOINT_DIR / "best_exp3.pt"

TRAIN_POSITIONS = ["NF"]
TEST_POSITIONS = ["MF"]
RECORDED_VAL_FRACTION = 0.2
RECORDED_SPLIT_BY_FILE = True

SPLIT_STRATEGY_POSITION = "position"
SPLIT_STRATEGY_CLASS = "class"
CLASS_SPLIT_TRAIN = [
    "minor_2nd", "major_2nd", "minor_3rd", "major_3rd",
    "perfect_4th", "tritone",
]
CLASS_SPLIT_VAL = ["perfect_5th", "minor_6th"]
CLASS_SPLIT_TEST = [
    "major_6th", "minor_7th", "major_7th", "perfect_8th",
]

# Synthetic test
SYNTHETIC_METADATA_CSV = PROJECT_ROOT / "dataset-preparation" / "synthetic-dataset" / "metadata.csv"
SYNTHETIC_DATASET_DIR = PROJECT_ROOT / "dataset-preparation" / "synthetic-dataset"
SYNTHETIC_TEST_INSTRUMENTS = ["violin", "flute"]

# Audio
SR = 22050
DURATION = 2.0
NUM_SAMPLES = int(SR * DURATION)

INTERVAL_CLASSES = [
    "minor_2nd", "major_2nd", "minor_3rd", "major_3rd",
    "perfect_4th", "tritone", "perfect_5th", "minor_6th",
    "major_6th", "minor_7th", "major_7th", "perfect_8th",
]
NUM_CLASSES = len(INTERVAL_CLASSES)
LABEL_TO_IDX = {name: i for i, name in enumerate(INTERVAL_CLASSES)}
IDX_TO_LABEL = {i: name for i, name in enumerate(INTERVAL_CLASSES)}

# Representations
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
    fmin=65.41,
)
HCQT_PARAMS = dict(
    hop_length=512,
    bins_per_octave=36,
    n_bins=252,
    fmin=65.41,
    harmonics=(1, 2, 3, 4, 5),
)

# Model: SpectrogramCNN for HCQT
CNN2D_CHANNELS = (32, 64, 128, 256)
CNN2D_KERNEL = 3
CLASSIFIER_DIM = 128
HCQT_IN_CHANNELS = 5

# Fine-tuning
BATCH_SIZE = 32
MAX_EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
PATIENCE = 15
T_MAX = 30
SEEDS = [42, 43, 44]

# SpecAugment
FREQ_MASK_PARAM = 15
TIME_MASK_PARAM = 10
NUM_FREQ_MASKS = 2
NUM_TIME_MASKS = 2

# Output
EXP3_RESULTS_TXT = RESULTS_DIR / "exp3_results.txt"
EXP3_RUN_LOG = RESULTS_DIR / "exp3_results.run.log"
EXP3_LOG = RESULTS_DIR / "exp3_run.log"
