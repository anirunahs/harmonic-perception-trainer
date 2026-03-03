"""
Feature extractors for the four representation conditions.

Each extractor takes a 1-D waveform numpy array (float32, mono, SR = 22 050)
and returns a numpy array ready for the corresponding CNN:

    mel  (1, 128, T)
    cqt  (1, F,   T)      F = n_bins (252)
    hcqt (H, F,   T)      H = len(harmonics) (5)
    raw  (1, N)           N = NUM_SAMPLES (44 100)
"""

from __future__ import annotations

import numpy as np
import librosa

from .config import (
    SR, NUM_SAMPLES,
    MEL_PARAMS, CQT_PARAMS, HCQT_PARAMS,
)


def extract_mel(audio: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR,
        n_fft=MEL_PARAMS["n_fft"],
        hop_length=MEL_PARAMS["hop_length"],
        n_mels=MEL_PARAMS["n_mels"],
        fmin=MEL_PARAMS["fmin"],
        fmax=MEL_PARAMS["fmax"],
    )
    S_db = np.log(S + 1e-6).astype(np.float32)
    return S_db[np.newaxis, ...]                   # (1, 128, T)


def extract_cqt(audio: np.ndarray) -> np.ndarray:
    C = np.abs(librosa.cqt(
        y=audio, sr=SR,
        hop_length=CQT_PARAMS["hop_length"],
        fmin=CQT_PARAMS["fmin"],
        n_bins=CQT_PARAMS["n_bins"],
        bins_per_octave=CQT_PARAMS["bins_per_octave"],
    ))
    C_db = np.log(C + 1e-6).astype(np.float32)
    return C_db[np.newaxis, ...]                   # (1, 252, T)


def _safe_cqt(audio: np.ndarray, fmin: float,
              n_bins: int, bins_per_octave: int,
              hop_length: int) -> np.ndarray:
    """CQT that clips n_bins to stay below Nyquist."""
    max_bins = int(bins_per_octave * np.log2((SR / 2) / fmin))
    actual = min(n_bins, max(1, max_bins))
    C = np.abs(librosa.cqt(
        y=audio, sr=SR,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=actual,
        bins_per_octave=bins_per_octave,
    ))
    if actual < n_bins:
        pad = np.zeros((n_bins - actual, C.shape[1]), dtype=C.dtype)
        C = np.concatenate([C, pad], axis=0)
    return C


def extract_hcqt(audio: np.ndarray) -> np.ndarray:
    p = HCQT_PARAMS
    layers = []
    for h in p["harmonics"]:
        C = _safe_cqt(
            audio,
            fmin=p["fmin"] * h,
            n_bins=p["n_bins"],
            bins_per_octave=p["bins_per_octave"],
            hop_length=p["hop_length"],
        )
        layers.append(C)
    hcqt = np.stack(layers, axis=0)                # (H, F, T)
    hcqt = np.log(hcqt + 1e-6).astype(np.float32)
    return hcqt                                    # (5, 252, T)


def extract_raw(audio: np.ndarray) -> np.ndarray:
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    elif len(audio) > NUM_SAMPLES:
        audio = audio[:NUM_SAMPLES]
    return audio.astype(np.float32)[np.newaxis, ...]  # (1, 44100)


EXTRACTORS = {
    "mel":  extract_mel,
    "cqt":  extract_cqt,
    "hcqt": extract_hcqt,
    "raw":  extract_raw,
}


def get_extractor(name: str):
    if name not in EXTRACTORS:
        raise ValueError(f"Unknown representation: {name}. "
                         f"Choose from {list(EXTRACTORS)}")
    return EXTRACTORS[name]
