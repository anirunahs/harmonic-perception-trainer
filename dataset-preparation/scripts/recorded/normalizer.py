"""
Audio normalization for recorded dataset.

Handles loading, LUFS normalization, bandpass filtering, and noise reduction.
"""

import logging

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import (
    SAMPLE_RATE,
    TARGET_LUFS,
    HIGHPASS_CUTOFF,
    LOWPASS_CUTOFF,
    FILTER_ORDER,
    PEAK_CEILING,
    NOISE_PROFILE_DURATION,
    NOISE_REDUCE_PROP,
    NOISE_REDUCE_STATIONARY,
)

logger = logging.getLogger(__name__)


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and convert to mono at the target sample rate."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def apply_bandpass(audio: np.ndarray, sr: int = SAMPLE_RATE,
                   low: int = HIGHPASS_CUTOFF,
                   high: int = LOWPASS_CUTOFF,
                   order: int = FILTER_ORDER) -> np.ndarray:
    """Apply a bandpass filter (Butterworth, zero-phase) to remove rumble and hiss."""
    nyquist = sr / 2.0
    lo = low / nyquist
    hi = high / nyquist

    lo = max(lo, 1e-5)
    hi = min(hi, 1.0 - 1e-5)

    if lo >= hi:
        return audio

    sos = butter(order, [lo, hi], btype='band', output='sos')
    return sosfiltfilt(sos, audio).astype(np.float32)


def normalize_lufs(audio: np.ndarray, sr: int = SAMPLE_RATE,
                   target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """LUFS normalization (EBU R128) with peak ceiling."""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if np.isfinite(loudness):
            audio = pyln.normalize.loudness(audio, loudness, target_lufs)
    except ImportError:
        logger.warning("pyloudnorm not installed — falling back to RMS normalization")
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** ((-18.0) / 20)
            audio = audio * (target_rms / rms)

    peak = np.max(np.abs(audio))
    if peak > PEAK_CEILING:
        audio = audio * (PEAK_CEILING / peak)

    return audio.astype(np.float32)


def reduce_noise(audio: np.ndarray, sr: int = SAMPLE_RATE,
                 noise_profile: np.ndarray = None,
                 prop_decrease: float = NOISE_REDUCE_PROP) -> np.ndarray:
    """Spectral noise reduction using noisereduce."""
    try:
        import noisereduce as nr
    except ImportError:
        logger.warning(
            "noisereduce not installed (pip install noisereduce) — skipping noise reduction"
        )
        return audio

    if noise_profile is None:
        profile_samples = int(NOISE_PROFILE_DURATION * sr)
        profile_samples = min(profile_samples, len(audio))
        noise_profile = audio[:profile_samples]

    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_profile,
        prop_decrease=prop_decrease,
        stationary=NOISE_REDUCE_STATIONARY,
    )
    return reduced.astype(np.float32)


def normalize_recording(path: str, sr: int = SAMPLE_RATE,
                        apply_nr: bool = True) -> np.ndarray:
    """Full normalization pipeline for a single long recording.

    Steps:
      1. Load at target SR (mono)
      2. DC offset removal
      3. Bandpass filter (80 Hz – 8 kHz)
      4. Noise reduction (spectral, from silence profile)
      5. LUFS normalization with peak ceiling

    Returns the processed audio array.
    """
    audio = load_audio(path, sr)
    logger.info(f"Loaded {path}: {len(audio)/sr:.1f}s, {len(audio)} samples")

    audio = audio - np.mean(audio)

    audio = apply_bandpass(audio, sr)

    if apply_nr:
        audio = reduce_noise(audio, sr)

    audio = normalize_lufs(audio, sr)

    return audio
