"""
Pitch-shift augmentation for recorded segments.

Generates augmented copies by shifting pitch ±1 semitone.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import librosa
import numpy as np

from .config import SAMPLE_RATE, AUGMENT_SEMITONES

logger = logging.getLogger(__name__)


@dataclass
class AugmentedSegment:
    """An augmented copy of an original segment."""
    audio: np.ndarray
    semitone_shift: int
    parent_onset_index: int


def pitch_shift_segment(audio: np.ndarray, semitones: int,
                        sr: int = SAMPLE_RATE) -> np.ndarray:
    """Shift the pitch of a 2-second segment by `semitones` semitones.

    Uses librosa's high-quality pitch shifter with scipy backend.
    """
    if semitones == 0:
        return audio.copy()

    shifted = librosa.effects.pitch_shift(
        audio, sr=sr, n_steps=float(semitones),
        bins_per_octave=12, res_type='soxr_hq',
    )
    return shifted.astype(np.float32)


def augment_segments(
    segments_with_reports: List[Tuple],
    sr: int = SAMPLE_RATE,
    shifts: List[int] = None,
) -> List[Tuple]:
    """Create pitch-shifted augmented copies of accepted segments.

    Args:
        segments_with_reports: list of (Segment, QualityReport) tuples
        sr: sample rate
        shifts: list of semitone shifts (default: [-1, +1])

    Returns:
        List of (AugmentedSegment, parent_Segment, shift) tuples.
    """
    if shifts is None:
        shifts = list(AUGMENT_SEMITONES)

    augmented: List[Tuple] = []
    total = len(segments_with_reports) * len(shifts)

    for i, (seg, _report) in enumerate(segments_with_reports):
        for shift in shifts:
            shifted_audio = pitch_shift_segment(seg.audio, shift, sr)
            aug = AugmentedSegment(
                audio=shifted_audio,
                semitone_shift=shift,
                parent_onset_index=seg.onset_index,
            )
            augmented.append((aug, seg, shift))

        if (i + 1) % 50 == 0 or (i + 1) == len(segments_with_reports):
            done = (i + 1) * len(shifts)
            logger.info(f"Augmentation progress: {done}/{total}")

    logger.info(f"Created {len(augmented)} augmented segments")
    return augmented
