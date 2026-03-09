"""
FFT-based numerical features for interval recognition baseline.

Matches backend/core/feature_extraction.py: magnitude spectrum 80–4000 Hz (one frame)
+ spectral stats + peak/harmonic features ~384 features.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

FFT_SIZE = 2048
MIN_FREQ = 80
MAX_FREQ = 4000
TARGET_LENGTH = 384


def _freq_range(sample_rate: int = 22050) -> tuple[int, int]:
    freqs = np.fft.fftfreq(FFT_SIZE, 1.0 / sample_rate)[: FFT_SIZE // 2 + 1]
    min_idx = int(np.argmax(freqs >= MIN_FREQ))
    max_idx = int(np.argmax(freqs >= MAX_FREQ))
    if max_idx == 0:
        max_idx = len(freqs)
    return min_idx, max_idx


def _prepare_audio(audio: np.ndarray) -> np.ndarray:
    """One frame: center 2048 samples (or pad/crop)."""
    if len(audio) < FFT_SIZE:
        audio = np.pad(audio.astype(np.float64), (0, FFT_SIZE - len(audio)))
    elif len(audio) > FFT_SIZE:
        start = (len(audio) - FFT_SIZE) // 2
        audio = audio[start : start + FFT_SIZE].astype(np.float64)
    return audio


def _extract_fft_magnitude(audio: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
    windowed = audio * np.hanning(len(audio))
    fft = np.fft.fft(windowed, n=FFT_SIZE)
    magnitude = np.abs(fft[: FFT_SIZE // 2 + 1])[min_idx:max_idx]
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    mn, mx = magnitude_db.min(), magnitude_db.max()
    if mx - mn < 1e-10:
        return np.zeros_like(magnitude_db)
    return ((magnitude_db - mn) / (mx - mn)).astype(np.float32)


def _find_peaks(magnitude: np.ndarray, prominence: float = 0.1) -> np.ndarray:
    if find_peaks is None:
        peaks = []
        m = magnitude
        for i in range(1, len(m) - 1):
            if m[i] > m[i - 1] and m[i] > m[i + 1] and m[i] > prominence * np.max(m):
                peaks.append(i)
            if len(peaks) >= 10:
                break
        return np.array(peaks)
    p, _ = find_peaks(magnitude, prominence=prominence * (np.max(magnitude) + 1e-10))
    return p[:10]


def _peak_features(magnitude: np.ndarray, freqs: np.ndarray) -> list:
    """Extract peak-based features."""
    features = []
    peak_indices = _find_peaks(magnitude)
    if len(peak_indices) == 0:
        return [0.0] * 15
    fundamental_freq = freqs[peak_indices[0]]
    features.append(float(fundamental_freq / MAX_FREQ))
    for i in range(1, min(5, len(peak_indices))):
        ratio = freqs[peak_indices[i]] / fundamental_freq if fundamental_freq > 0 else 0.0
        features.append(float(ratio))
    while len(features) < 6:
        features.append(0.0)
    max_mag = np.max(magnitude) + 1e-10
    for i in range(min(5, len(peak_indices))):
        features.append(float(magnitude[peak_indices[i]] / max_mag))
    while len(features) < 11:
        features.append(0.0)
    for i in range(1, min(5, len(peak_indices))):
        interval = (freqs[peak_indices[i]] - freqs[peak_indices[i - 1]]) / MAX_FREQ
        features.append(float(interval))
    while len(features) < 15:
        features.append(0.0)
    return features[:15]


def extract_fft_features(
    audio: np.ndarray,
    sr: int = 22050,
) -> np.ndarray:
    """Extract FFT features."""
    min_idx, max_idx = _freq_range(sr)
    audio = _prepare_audio(audio)
    magnitude_fft = _extract_fft_magnitude(audio, min_idx, max_idx)
    freqs = np.linspace(MIN_FREQ, MAX_FREQ, len(magnitude_fft))
    magnitude = magnitude_fft.astype(np.float64) + 1e-10

    spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
    features = [
        spectral_centroid / MAX_FREQ,
        np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-10)) / MAX_FREQ,
    ]
    cumsum = np.cumsum(magnitude)
    rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
    features.append(float(freqs[rolloff_idx[0]] / MAX_FREQ) if len(rolloff_idx) > 0 else 1.0)
    features.append(float(np.sqrt(np.mean(audio ** 2))))
    features.append(float(np.mean(np.abs(np.diff(np.sign(audio))))))

    features.extend(_peak_features(magnitude, freqs))

    additional = np.array(features, dtype=np.float32)
    n_fft = max_idx - min_idx

    if len(magnitude_fft) > n_fft:
        magnitude_fft = magnitude_fft[:n_fft]
    elif len(magnitude_fft) < n_fft:
        magnitude_fft = np.pad(magnitude_fft, (0, n_fft - len(magnitude_fft)))
    n_extra = TARGET_LENGTH - len(magnitude_fft) - len(additional)
    if n_extra > 0:
        additional = np.pad(additional, (0, n_extra))
    elif n_extra < 0:
        additional = additional[: TARGET_LENGTH - len(magnitude_fft)]
    out = np.concatenate([magnitude_fft, additional]).astype(np.float32)
    return out[:TARGET_LENGTH]


def get_fft_feature_dim() -> int:
    return TARGET_LENGTH
