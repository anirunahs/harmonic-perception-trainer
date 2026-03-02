"""
Configuration for recorded (microphone) dataset processing pipeline.

Defines interval mapping, file naming conventions, audio parameters,
quality thresholds, and split strategies for the recorded dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SAMPLE_RATE = 22050
SEGMENT_DURATION = 2.0
EXPECTED_INTERVALS_PER_FILE = 60

# Audio normalization
TARGET_LUFS = -23.0
HIGHPASS_CUTOFF = 80       # Hz — remove low-frequency rumble
LOWPASS_CUTOFF = 8000      # Hz — remove high-frequency artifacts
FILTER_ORDER = 4
PEAK_CEILING = 0.95        # prevent digital clipping after normalization

# Noise reduction
NOISE_PROFILE_DURATION = 0.5   # seconds of silence to sample for noise profile
NOISE_REDUCE_PROP = 0.8        # strength of noise reduction (0.0–1.0)
NOISE_REDUCE_STATIONARY = True

# Onset detection
ONSET_HOP_LENGTH = 512
ONSET_BACKTRACK = True
ONSET_PRE_MAX = 20
ONSET_POST_MAX = 20
ONSET_PRE_AVG = 100
ONSET_POST_AVG = 100
ONSET_DELTA = 0.2
ONSET_WAIT = 10
ONSET_MIN_INTERVAL_SEC = 1.5  # minimum gap between consecutive onsets

# Quality filter thresholds
QUALITY_MIN_RMS_DB = -55.0          # segments quieter than this are likely silence
QUALITY_MAX_SILENCE_RATIO = 0.70    # reject if >70% of segment is near-silent
QUALITY_MAX_CLICK_SCORE = 0.15      # transient / click detection threshold
QUALITY_MIN_SPECTRAL_ROLLOFF = 200  # Hz — reject if no meaningful harmonic content
QUALITY_MAX_SPECTRAL_FLATNESS = 0.6 # reject noise-like segments (white noise → 1.0)
QUALITY_MAX_ZCR = 0.25              # reject very noisy / hissy segments

# Augmentation
AUGMENT_SEMITONES = [-1, 1]

# File naming: raw recordings follow this pattern
FILENAME_TO_INTERVAL = {
    'min2':  'minor_2nd',
    'maj2':  'major_2nd',
    'min3':  'minor_3rd',
    'maj3':  'major_3rd',
    'perf4': 'perfect_4th',
    'trit':  'tritone',
    'tritone': 'tritone',
    'perf5': 'perfect_5th',
    'min6':  'minor_6th',
    'maj6':  'major_6th',
    'min7':  'minor_7th',
    'maj7':  'major_7th',
    'perf8': 'perfect_8th',
}

FILENAME_TO_DYNAMICS = {
    'soft': 'piano',
    'hard': 'forte',
}

INTERVAL_NAMES = list(FILENAME_TO_INTERVAL.values())
INTERVAL_SEMITONES = {
    'minor_2nd': 1,
    'major_2nd': 2,
    'minor_3rd': 3,
    'major_3rd': 4,
    'perfect_4th': 5,
    'tritone': 6,
    'perfect_5th': 7,
    'minor_6th': 8,
    'major_6th': 9,
    'minor_7th': 10,
    'major_7th': 11,
    'perfect_8th': 12,
}


@dataclass(frozen=True)
class RecordingPosition:
    """Microphone position metadata."""
    name: str
    folder: str
    description: str
    distance_cm: int
    needs_noise_reduction: bool


# NF = Near Field  — on the piano lid, ~10–20 cm.  Already noise-reduced in Audacity.
# MF = Mid Field   — table ~80 cm from piano body.  Raw recording, needs NR.
# FF = Far Field   — couch ~1.5 m from piano body.  Not recommended (loud + reverb).
POSITIONS = {
    'NF': RecordingPosition('NF', 'NF', 'Near Field (~0.1–0.2 m, on lid)',  15, needs_noise_reduction=False),
    'MF': RecordingPosition('MF', 'MF', 'Mid Field (~0.8 m, table)',        80, needs_noise_reduction=True),
    'FF': RecordingPosition('FF', 'FF', 'Far Field (~1.5 m, couch)',       150, needs_noise_reduction=True),
}

DEFAULT_POSITIONS = ['NF', 'MF']


SPLIT_STRATEGIES = {
    'position': {
        'train': ['NF'],
        'test':  ['MF'],
    },
    'class': {
        'train': [
            'minor_2nd', 'major_2nd', 'minor_3rd', 'major_3rd',
            'perfect_4th', 'tritone',
        ],
        'val': ['perfect_5th', 'minor_6th'],
        'test': [
            'major_6th', 'minor_7th', 'major_7th', 'perfect_8th',
        ],
    },
}


DEFAULT_RAW_DIR = Path('dataset-preparation/raw-recordings')
DEFAULT_OUTPUT_DIR = Path('dataset-preparation/processed-segments')


def parse_recording_filename(filename: str) -> Tuple[str, str]:
    """Parse interval and dynamics from a recording filename.

    Expected patterns: 'min2_soft.wav', 'perf5_hard.wav', etc.
    Returns (interval_name, dynamics_name) or raises ValueError.
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) < 2:
        raise ValueError(f"Cannot parse filename: {filename}")

    interval_code = parts[0]
    dynamics_code = parts[1]

    interval = FILENAME_TO_INTERVAL.get(interval_code)
    dynamics = FILENAME_TO_DYNAMICS.get(dynamics_code)

    if not interval:
        raise ValueError(f"Unknown interval code '{interval_code}' in {filename}")
    if not dynamics:
        raise ValueError(f"Unknown dynamics code '{dynamics_code}' in {filename}")

    return interval, dynamics
