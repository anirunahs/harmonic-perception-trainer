"""
Experiment 4 — Interpretability and FFT baseline.

RQ: Do models focus on physically grounded frequency regions (notes, overtones)
    or on artifacts? Comparison of HCQT CNN vs FFT-based MLP.

Design: Grad-CAM for best HCQT CNN, t-SNE/UMAP embeddings, confusion matrix
    analysis, FFT numerical-feature MLP (synthetic / recorded / combined).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Data
SYNTHETIC_METADATA = PROJECT_ROOT / "dataset-preparation" / "synthetic-dataset" / "metadata.csv"
SYNTHETIC_DIR = PROJECT_ROOT / "dataset-preparation" / "synthetic-dataset"
RECORDED_METADATA = PROJECT_ROOT / "dataset-preparation" / "processed-segments" / "metadata.csv"
RECORDED_DIR = PROJECT_ROOT / "dataset-preparation" / "processed-segments"
# Checkpoints
EXP1_DIR = PROJECT_ROOT / "experiments" / "exp1_representations"
EXP3_DIR = PROJECT_ROOT / "experiments" / "exp3_microphone"
BEST_HCQT_EXP1 = EXP1_DIR / ".cache" / "checkpoints" / "best_hcqt.pt"
BEST_HCQT_EXP3 = EXP3_DIR / ".cache" / "checkpoints" / "best_exp3.pt"
# Output
RESULTS_DIR = PROJECT_ROOT / "ztu" / "docs" / "experiments"
EXP4_RESULTS_TXT = RESULTS_DIR / "exp4_results.txt"
EXP4_RUN_LOG = RESULTS_DIR / "exp4_run.log"
EXP4_FIGURES_DIR = RESULTS_DIR / "exp4_figures"
EXP4_GRADCAM_DIR = RESULTS_DIR / "exp4_figures" / "gradcam"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
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

# Synthetic split
SPLIT_INSTRUMENTS = {
    "train": ["piano", "guitar_nylon", "guitar_steel", "cello", "church_organ", "trumpet"],
    "val": ["clarinet", "marimba"],
    "test": ["violin", "flute"],
}
TRAIN_DEV_FRACTION = 0.15

# Recorded split
TRAIN_POSITIONS = ["NF"]
TEST_POSITIONS = ["MF"]
RECORDED_VAL_FRACTION = 0.2
RECORDED_SPLIT_BY_FILE = True

# HCQT
HCQT_PARAMS = dict(
    hop_length=512,
    bins_per_octave=36,
    n_bins=252,
    fmin=65.41,
    harmonics=(1, 2, 3, 4, 5),
)
CNN2D_CHANNELS = (32, 64, 128, 256)
CNN2D_KERNEL = 3
CLASSIFIER_DIM = 128
HCQT_IN_CHANNELS = 5
DROPOUT = 0.3

# FFT feature extraction
N_FFT = 2048
FFT_FEATURE_DIM = 384

# MLP for FFT
MLP_HIDDEN = (768, 384, 192, 96)
MLP_DROPOUT = 0.25
BATCH_SIZE = 32
MAX_EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 22
FFT_LR_T_MAX = 50
FFT_LABEL_SMOOTHING = 0.1
SEED = 42

# Visualization
GRADCAM_N_SAMPLES_PER_CLASS = 2
TSNE_PERPLEXITY = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
