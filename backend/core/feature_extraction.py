import numpy as np
import librosa
from scipy.signal import find_peaks
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class FFTFeatureExtractor:
    """Виділення ознак"""

    def __init__(self, fft_size: int = 2048, sample_rate: int = 22050):
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.freq_bins = fft_size // 2 + 1
        
        self.min_freq = 80
        self.max_freq = 4000
        self.freq_range = self._calculate_freq_range()
        
        self.target_length = 384
    
    def _calculate_freq_range(self) -> Tuple[int, int]:
        """Розрахунок індексів частотного діапазону"""
        freqs = np.fft.fftfreq(self.fft_size, 1/self.sample_rate)[:self.freq_bins]
        min_idx = np.argmax(freqs >= self.min_freq)
        max_idx = np.argmax(freqs >= self.max_freq)
        if max_idx == 0:
            max_idx = len(freqs)
        return min_idx, max_idx

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Виділення ознак"""
        try:
            audio = self._prepare_audio_identical(audio)
            
            fft_features = self._extract_fft_features_identical(audio)
            
            additional_features = self._extract_additional_features_identical(audio, fft_features)
            
            all_features = np.concatenate([fft_features, additional_features])
            
            logger.debug(f"Виділено {len(all_features)} ознак")
            return all_features
            
        except Exception as e:
            logger.error(f"Помилка виділення ознак: {e}")
            return np.zeros(self.target_length)
    
    def _prepare_audio_identical(self, audio: np.ndarray) -> np.ndarray:
        """Підготовка аудіо"""
        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)))
        elif len(audio) > self.fft_size:
            start = (len(audio) - self.fft_size) // 2
            audio = audio[start:start + self.fft_size]
        
        return audio
    
    def _extract_fft_features_identical(self, audio: np.ndarray) -> np.ndarray:
        """FFT ознаки"""

        windowed_audio = audio * np.hanning(len(audio))
        
        fft = np.fft.fft(windowed_audio, n=self.fft_size)
        magnitude = np.abs(fft[:self.freq_bins])
        
        min_idx, max_idx = self.freq_range
        magnitude = magnitude[min_idx:max_idx]
        
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        magnitude_normalized = (magnitude_db - np.min(magnitude_db)) / (np.max(magnitude_db) - np.min(magnitude_db) + 1e-10)
        
        return magnitude_normalized
    
    def _extract_additional_features_identical(self, audio: np.ndarray, magnitude_fft: np.ndarray) -> np.ndarray:
        """Додаткові ознаки"""
        features = []
        
        freqs = np.linspace(self.min_freq, self.max_freq, len(magnitude_fft))
        magnitude = magnitude_fft
        
        # Спектральний центроїд
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        features.append(spectral_centroid / self.max_freq)
        
        # Спектральна ширина
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-10))
        features.append(spectral_bandwidth / self.max_freq)
        
        # Спектральний rolloff
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            rolloff_freq = freqs[rolloff_idx[0]]
            features.append(rolloff_freq / self.max_freq)
        else:
            features.append(1.0)
        
        # RMS енергія
        rms = np.sqrt(np.mean(audio ** 2))
        features.append(rms)
        
        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        features.append(zcr)
        
        # Піки та гармоніки (15 ознак)
        peak_features = self._calculate_peak_features_identical(magnitude, freqs)
        features.extend(peak_features)
        
        return np.array(features)
    
    def _find_spectral_peaks_identical(self, magnitude: np.ndarray, prominence: float = 0.1):
        """Знаходження піків"""
        try:
            peaks, _ = find_peaks(magnitude, prominence=prominence * np.max(magnitude))
            return peaks[:10]
        except ImportError:
            peaks = []
            for i in range(1, len(magnitude) - 1):
                if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                    if magnitude[i] > prominence * np.max(magnitude):
                        peaks.append(i)
                if len(peaks) >= 10:
                    break
            return peaks
    
    def _calculate_peak_features_identical(self, magnitude: np.ndarray, freqs: np.ndarray) -> list:
        """Розрахунок ознак піків"""
        features = []
        
        peak_indices = self._find_spectral_peaks_identical(magnitude)
        
        if len(peak_indices) == 0:
            return [0.0] * 15
        
        fundamental_freq = freqs[peak_indices[0]] if len(peak_indices) > 0 else 1.0
        features.append(fundamental_freq / self.max_freq)
        
        for i in range(1, min(5, len(peak_indices))):
            ratio = freqs[peak_indices[i]] / fundamental_freq if fundamental_freq > 0 else 0.0
            features.append(ratio)
        
        while len(features) < 6:
            features.append(0.0)
        
        max_magnitude = np.max(magnitude)
        for i in range(min(5, len(peak_indices))):
            amplitude = magnitude[peak_indices[i]] / max_magnitude
            features.append(amplitude)
        
        while len(features) < 11:
            features.append(0.0)
        
        for i in range(1, min(5, len(peak_indices))):
            interval = (freqs[peak_indices[i]] - freqs[peak_indices[i-1]]) / self.max_freq
            features.append(interval)
        
        while len(features) < 15:
            features.append(0.0)
        
        return features[:15]
    
    def get_expected_feature_count(self) -> int:
        """Очікувана кількість ознак"""
        min_idx, max_idx = self.freq_range
        fft_features = max_idx - min_idx
        additional_features = 20
        return fft_features + additional_features