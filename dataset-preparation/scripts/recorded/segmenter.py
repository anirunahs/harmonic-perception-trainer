"""
Onset detection and 2-second segment extraction from long recordings.

Uses librosa onset detection with energy-based validation to reliably
locate interval boundaries, then extracts fixed-length segments.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import librosa
import numpy as np

from .config import (
    SAMPLE_RATE,
    SEGMENT_DURATION,
    ONSET_HOP_LENGTH,
    ONSET_BACKTRACK,
    ONSET_PRE_MAX,
    ONSET_POST_MAX,
    ONSET_PRE_AVG,
    ONSET_POST_AVG,
    ONSET_DELTA,
    ONSET_WAIT,
    ONSET_MIN_INTERVAL_SEC,
)

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A single extracted audio segment with metadata."""
    audio: np.ndarray
    start_sec: float
    end_sec: float
    onset_index: int
    rms_db: float


def detect_onsets(audio: np.ndarray, sr: int = SAMPLE_RATE,
                  min_interval_sec: float = ONSET_MIN_INTERVAL_SEC) -> np.ndarray:
    """Detect note onsets in a long recording.

    Returns an array of onset times in seconds, filtered so that consecutive
    onsets are at least `min_interval_sec` apart.
    """
    onset_times = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=ONSET_HOP_LENGTH,
        backtrack=ONSET_BACKTRACK,
        units='time',
        pre_max=ONSET_PRE_MAX,
        post_max=ONSET_POST_MAX,
        pre_avg=ONSET_PRE_AVG,
        post_avg=ONSET_POST_AVG,
        delta=ONSET_DELTA,
        wait=ONSET_WAIT,
    )

    if len(onset_times) == 0:
        logger.warning("No onsets detected — trying with lower delta")
        onset_times = librosa.onset.onset_detect(
            y=audio, sr=sr,
            hop_length=ONSET_HOP_LENGTH,
            backtrack=ONSET_BACKTRACK,
            units='time',
            delta=ONSET_DELTA * 0.5,
            wait=ONSET_WAIT,
        )

    filtered = _filter_onsets(onset_times, min_interval_sec)
    logger.info(f"Detected {len(onset_times)} raw onsets → {len(filtered)} after filtering")
    return filtered


def _filter_onsets(onsets: np.ndarray,
                   min_gap: float) -> np.ndarray:
    """Remove onsets that are too close together."""
    if len(onsets) == 0:
        return onsets

    kept = [onsets[0]]
    for t in onsets[1:]:
        if t - kept[-1] >= min_gap:
            kept.append(t)
    return np.array(kept)


def extract_segments(audio: np.ndarray,
                     onsets: np.ndarray,
                     sr: int = SAMPLE_RATE,
                     duration: float = SEGMENT_DURATION,
                     pre_onset_ms: float = 30.0) -> List[Segment]:
    """Extract fixed-length segments starting slightly before each onset.

    A small pre-onset margin (default 30 ms) captures the attack transient.
    Segments that would extend past the end of the recording are zero-padded.
    """
    segment_samples = int(duration * sr)
    pre_samples = int(pre_onset_ms / 1000.0 * sr)
    segments: List[Segment] = []

    for idx, onset_sec in enumerate(onsets):
        onset_sample = int(onset_sec * sr)
        start = max(0, onset_sample - pre_samples)
        end = start + segment_samples

        if start >= len(audio):
            continue

        if end <= len(audio):
            seg_audio = audio[start:end].copy()
        else:
            seg_audio = np.zeros(segment_samples, dtype=np.float32)
            available = len(audio) - start
            seg_audio[:available] = audio[start:start + available]

        rms = np.sqrt(np.mean(seg_audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        segments.append(Segment(
            audio=seg_audio,
            start_sec=start / sr,
            end_sec=(start + segment_samples) / sr,
            onset_index=idx,
            rms_db=rms_db,
        ))

    logger.info(f"Extracted {len(segments)} segments of {duration}s each")
    return segments


def segment_recording(audio: np.ndarray,
                      sr: int = SAMPLE_RATE,
                      expected_count: Optional[int] = None) -> List[Segment]:
    """Full segmentation pipeline: detect onsets, then extract 2s segments.

    If `expected_count` is provided, logs a warning when the detected count
    differs significantly.
    """
    onsets = detect_onsets(audio, sr)

    if expected_count is not None:
        diff = abs(len(onsets) - expected_count)
        if diff > expected_count * 0.1:
            logger.warning(
                f"Expected ~{expected_count} onsets but detected {len(onsets)} "
                f"(difference: {diff})"
            )

    return extract_segments(audio, onsets, sr)
