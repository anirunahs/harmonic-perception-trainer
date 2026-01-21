"""
FFT-based feature extraction strategy.

Pattern: Strategy (Behavioral) - Concrete Strategy
"""

import numpy as np
from scipy.signal import find_peaks
import logging
from typing import Tuple

from .base import FeatureExtractionStrategy

logger = logging.getLogger(__name__)


class FFTStrategy(FeatureExtractionStrategy):
    """
    FFT-based feature extraction strategy.
    
    Extracts features using Fast Fourier Transform:
    - Magnitude spectrum in frequency range 80-4000 Hz
    - Spectral centroid, bandwidth, rolloff
    - RMS energy and zero-crossing rate
    - Peak frequencies and harmonic ratios
    
    This is the production strategy used for interval classification.
    """
    
    DEFAULT_FFT_SIZE = 2048
    DEFAULT_SAMPLE_RATE = 22050
    MIN_FREQ = 80
    MAX_FREQ = 4000
    
    def __init__(
        self,
        fft_size: int = DEFAULT_FFT_SIZE,
        sample_rate: int = DEFAULT_SAMPLE_RATE
    ):
        """
        Initialize FFT strategy.
        
        Args:
            fft_size: FFT window size (default: 2048)
            sample_rate: Expected sample rate (default: 22050)
        """
        self._fft_size = fft_size
        self._sample_rate = sample_rate
        self._freq_bins = fft_size // 2 + 1
        self._freq_range = self._calculate_freq_range()
        self._target_length = 384
    
    def _calculate_freq_range(self) -> Tuple[int, int]:
        """Calculate frequency range indices."""
        freqs = np.fft.fftfreq(self._fft_size, 1/self._sample_rate)[:self._freq_bins]
        min_idx = np.argmax(freqs >= self.MIN_FREQ)
        max_idx = np.argmax(freqs >= self.MAX_FREQ)
        if max_idx == 0:
            max_idx = len(freqs)
        return min_idx, max_idx
    
    def extract_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract FFT-based features from audio.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate (used for validation)
            
        Returns:
            Feature vector
        """
        try:
            # Prepare audio
            audio = self._prepare_audio(audio)
            
            # Extract FFT features
            fft_features = self._extract_fft_features(audio)
            
            # Extract additional features
            additional_features = self._extract_additional_features(audio, fft_features)
            
            # Combine
            all_features = np.concatenate([fft_features, additional_features])
            
            logger.debug(f"FFTStrategy: extracted {len(all_features)} features")
            return all_features
            
        except Exception as e:
            logger.error(f"FFTStrategy: feature extraction error: {e}")
            return np.zeros(self._target_length)
    
    def get_name(self) -> str:
        """Get strategy name."""
        return "FFT-based"
    
    def get_feature_count(self) -> int:
        """Get expected feature count."""
        min_idx, max_idx = self._freq_range
        fft_features = max_idx - min_idx
        additional_features = 20
        return fft_features + additional_features
    
    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Prepare audio for FFT."""
        if len(audio) < self._fft_size:
            audio = np.pad(audio, (0, self._fft_size - len(audio)))
        elif len(audio) > self._fft_size:
            start = (len(audio) - self._fft_size) // 2
            audio = audio[start:start + self._fft_size]
        return audio
    
    def _extract_fft_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract FFT magnitude features."""
        # Apply Hanning window
        windowed_audio = audio * np.hanning(len(audio))
        
        # Compute FFT
        fft = np.fft.fft(windowed_audio, n=self._fft_size)
        magnitude = np.abs(fft[:self._freq_bins])
        
        # Limit to frequency range
        min_idx, max_idx = self._freq_range
        magnitude = magnitude[min_idx:max_idx]
        
        # Convert to dB and normalize
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        magnitude_normalized = (magnitude_db - np.min(magnitude_db)) / \
                              (np.max(magnitude_db) - np.min(magnitude_db) + 1e-10)
        
        return magnitude_normalized
    
    def _extract_additional_features(
        self,
        audio: np.ndarray,
        magnitude_fft: np.ndarray
    ) -> np.ndarray:
        """Extract additional spectral features."""
        features = []
        
        freqs = np.linspace(self.MIN_FREQ, self.MAX_FREQ, len(magnitude_fft))
        magnitude = magnitude_fft
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        features.append(spectral_centroid / self.MAX_FREQ)
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / 
            (np.sum(magnitude) + 1e-10)
        )
        features.append(spectral_bandwidth / self.MAX_FREQ)
        
        # Spectral rolloff
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            rolloff_freq = freqs[rolloff_idx[0]]
            features.append(rolloff_freq / self.MAX_FREQ)
        else:
            features.append(1.0)
        
        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        features.append(rms)
        
        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        features.append(zcr)
        
        # Peak features (15)
        peak_features = self._calculate_peak_features(magnitude, freqs)
        features.extend(peak_features)
        
        return np.array(features)
    
    def _find_spectral_peaks(
        self,
        magnitude: np.ndarray,
        prominence: float = 0.1
    ) -> list:
        """Find spectral peaks."""
        try:
            peaks, _ = find_peaks(magnitude, prominence=prominence * np.max(magnitude))
            return list(peaks[:10])
        except Exception:
            # Fallback manual peak detection
            peaks = []
            for i in range(1, len(magnitude) - 1):
                if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                    if magnitude[i] > prominence * np.max(magnitude):
                        peaks.append(i)
                if len(peaks) >= 10:
                    break
            return peaks
    
    def _calculate_peak_features(
        self,
        magnitude: np.ndarray,
        freqs: np.ndarray
    ) -> list:
        """Calculate peak-based features."""
        features = []
        
        peak_indices = self._find_spectral_peaks(magnitude)
        
        if len(peak_indices) == 0:
            return [0.0] * 15
        
        # Fundamental frequency
        fundamental_freq = freqs[peak_indices[0]] if len(peak_indices) > 0 else 1.0
        features.append(fundamental_freq / self.MAX_FREQ)
        
        # Frequency ratios (5)
        for i in range(1, min(5, len(peak_indices))):
            ratio = freqs[peak_indices[i]] / fundamental_freq if fundamental_freq > 0 else 0.0
            features.append(ratio)
        while len(features) < 6:
            features.append(0.0)
        
        # Amplitude ratios (5)
        max_magnitude = np.max(magnitude)
        for i in range(min(5, len(peak_indices))):
            amplitude = magnitude[peak_indices[i]] / max_magnitude
            features.append(amplitude)
        while len(features) < 11:
            features.append(0.0)
        
        # Frequency intervals (4)
        for i in range(1, min(5, len(peak_indices))):
            interval = (freqs[peak_indices[i]] - freqs[peak_indices[i-1]]) / self.MAX_FREQ
            features.append(interval)
        while len(features) < 15:
            features.append(0.0)
        
        return features[:15]
