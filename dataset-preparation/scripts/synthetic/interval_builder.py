"""
Interval builder: combines two note samples into a musical interval.

"""

import numpy as np

from .config import (
    SAMPLE_RATE,
    SEGMENT_DURATION,
    FADE_IN_MS,
    FADE_OUT_MS,
    PEAK_NORMALIZE_DB,
)


class IntervalBuilder:
    """Builds 2-second interval audio clips from individual note samples."""

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        duration: float = SEGMENT_DURATION,
    ):
        self.sr = sr
        self.duration = duration
        self.target_samples = int(sr * duration)


    def build_harmonic(
        self, note1: np.ndarray, note2: np.ndarray,
    ) -> np.ndarray:
        """Two notes played simultaneously, mixed and normalised."""
        n1 = self._fit_to_length(note1, self.target_samples)
        n2 = self._fit_to_length(note2, self.target_samples)

        combined = n1 + n2
        combined = self._apply_fade(combined)
        combined = self._peak_normalize(combined)
        return combined

    def build_melodic(
        self, note1: np.ndarray, note2: np.ndarray,
        gap_ms: float = 100,
    ) -> np.ndarray:
        """Two notes played sequentially with a short gap."""
        gap_samples = int(self.sr * gap_ms / 1000)
        note_samples = (self.target_samples - gap_samples) // 2

        n1 = self._fit_to_length(note1, note_samples)
        n2 = self._fit_to_length(note2, note_samples)

        n1 = self._apply_fade(n1)
        n2 = self._apply_fade(n2)

        gap = np.zeros(gap_samples, dtype=np.float32)
        combined = np.concatenate([n1, gap, n2])
        combined = self._fit_to_length(combined, self.target_samples)
        combined = self._peak_normalize(combined)
        return combined

    # Helpers

    @staticmethod
    def _fit_to_length(audio: np.ndarray, target_length: int) -> np.ndarray:
        """Trim or zero-pad to exact target length."""
        if len(audio) >= target_length:
            return audio[:target_length].copy()
        return np.pad(audio, (0, target_length - len(audio)))

    def _apply_fade(self, audio: np.ndarray) -> np.ndarray:
        """Smooth fade-in / fade-out envelope to eliminate clicks."""
        audio = audio.copy()
        fade_in = int(self.sr * FADE_IN_MS / 1000)
        fade_out = int(self.sr * FADE_OUT_MS / 1000)

        if 0 < fade_in < len(audio):
            audio[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
        if 0 < fade_out < len(audio):
            audio[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
        return audio

    @staticmethod
    def _peak_normalize(audio: np.ndarray) -> np.ndarray:
        """Scale so the absolute peak equals PEAK_NORMALIZE_DB."""
        peak = np.max(np.abs(audio))
        if peak < 1e-10:
            return audio
        target = 10 ** (PEAK_NORMALIZE_DB / 20)
        return audio * (target / peak)
