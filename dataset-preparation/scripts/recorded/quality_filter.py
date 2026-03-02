"""
Segment quality analysis and filtering.

Detects clicks, excessive noise, silence-heavy segments, and other
artifacts that make a sample unsuitable for training. Each segment
receives a quality report; the pipeline can then accept or reject it
based on configurable thresholds.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import librosa
import numpy as np

from .config import (
    SAMPLE_RATE,
    QUALITY_MIN_RMS_DB,
    QUALITY_MAX_SILENCE_RATIO,
    QUALITY_MAX_CLICK_SCORE,
    QUALITY_MIN_SPECTRAL_ROLLOFF,
    QUALITY_MAX_SPECTRAL_FLATNESS,
    QUALITY_MAX_ZCR,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Quality metrics for a single segment."""
    rms_db: float
    peak_db: float
    silence_ratio: float
    click_score: float
    spectral_rolloff: float
    spectral_flatness: float
    zcr: float
    issues: List[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        return len(self.issues) == 0


def analyze_segment(audio: np.ndarray, sr: int = SAMPLE_RATE) -> QualityReport:
    """Compute quality metrics for a single segment and flag issues."""
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    rms_db = 20 * np.log10(rms + 1e-10)
    peak_db = 20 * np.log10(peak + 1e-10)

    silence_thresh = max(peak * 0.02, 1e-4)
    silence_ratio = float(np.mean(np.abs(audio) < silence_thresh))

    click_score = _detect_clicks(audio, sr)

    rolloff = float(np.mean(
        librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
    ))

    flatness = float(np.mean(
        librosa.feature.spectral_flatness(y=audio)[0]
    ))

    zcr = float(np.mean(
        librosa.feature.zero_crossing_rate(audio)[0]
    ))

    issues: List[str] = []

    if rms_db < QUALITY_MIN_RMS_DB:
        issues.append('too_quiet')
    if silence_ratio > QUALITY_MAX_SILENCE_RATIO:
        issues.append('too_much_silence')
    if click_score > QUALITY_MAX_CLICK_SCORE:
        issues.append('click_detected')
    if rolloff < QUALITY_MIN_SPECTRAL_ROLLOFF:
        issues.append('no_harmonic_content')
    if flatness > QUALITY_MAX_SPECTRAL_FLATNESS:
        issues.append('noise_like')
    if zcr > QUALITY_MAX_ZCR:
        issues.append('high_zcr')

    return QualityReport(
        rms_db=rms_db,
        peak_db=peak_db,
        silence_ratio=silence_ratio,
        click_score=click_score,
        spectral_rolloff=rolloff,
        spectral_flatness=flatness,
        zcr=zcr,
        issues=issues,
    )


def _detect_clicks(audio: np.ndarray, sr: int) -> float:
    """Detect clicks / transient artifacts in a segment.

    Uses the ratio of short-window energy spikes to median energy.
    A high ratio suggests an impulsive artifact (click, pop, rustle)
    that is not part of the musical signal.
    """
    hop = 256
    frame_length = 512
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop)
    frame_energy = np.sum(frames ** 2, axis=0)

    if len(frame_energy) < 4:
        return 0.0

    median_energy = np.median(frame_energy)
    if median_energy < 1e-12:
        return 0.0

    ratio = frame_energy / (median_energy + 1e-12)

    first_quarter = len(frame_energy) // 4
    non_attack = ratio[first_quarter:]

    if len(non_attack) == 0:
        return 0.0

    spike_threshold = 8.0
    spike_count = np.sum(non_attack > spike_threshold)
    click_score = spike_count / len(non_attack)

    return float(click_score)


def filter_segments(segments, sr: int = SAMPLE_RATE):
    """Analyze and partition segments into accepted and rejected lists.

    Returns (accepted, rejected) where each element is a tuple of
    (segment, quality_report).
    """
    accepted = []
    rejected = []

    for seg in segments:
        report = analyze_segment(seg.audio, sr)
        if report.is_acceptable:
            accepted.append((seg, report))
        else:
            rejected.append((seg, report))
            logger.debug(
                f"Rejected segment {seg.onset_index}: {', '.join(report.issues)} "
                f"(RMS={report.rms_db:.1f}dB, click={report.click_score:.3f})"
            )

    n_total = len(segments)
    n_rejected = len(rejected)
    logger.info(
        f"Quality filter: {len(accepted)}/{n_total} accepted, "
        f"{n_rejected} rejected"
    )

    if n_rejected > 0:
        issue_counts: dict = {}
        for _, r in rejected:
            for issue in r.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {issue}: {count}")

    return accepted, rejected
