import numpy as np
import librosa
from scipy.signal import find_peaks
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class FFTFeatureExtractor:
    """Клас для виділення FFT ознак з аудіо"""
    
    def __init__(self, fft_size: int = 2048, sample_rate: int = 22050):
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.freq_bins = fft_size // 2 + 1
        
        self.min_freq = 80
        self.max_freq = 4000
        self.freq_range = self._calculate_freq_range()
        
    def _calculate_freq_range(self) -> Tuple[int, int]:
        """Розрахунок індексів частотного діапазону"""
        freqs = np.fft.fftfreq(self.fft_size, 1/self.sample_rate)[:self.freq_bins]
        min_idx = np.argmax(freqs >= self.min_freq)
        max_idx = np.argmax(freqs >= self.max_freq)
        if max_idx == 0:
            max_idx = len(freqs)
        return min_idx, max_idx
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Виділення повного набору ознак з аудіо"""
        try:
            audio = self._prepare_audio(audio)
            
            fft_features = self._extract_fft_features(audio)
            
            musical_features = self._extract_musical_features(audio)
            
            spectral_features = self._extract_spectral_features(audio)
            
            harmonic_features = self._extract_harmonic_features(audio)
            
            all_features = np.concatenate([
                fft_features,
                musical_features,
                spectral_features,
                harmonic_features
            ])
            
            logger.debug(f"Виділено {len(all_features)} ознак")
            return all_features
            
        except Exception as e:
            logger.error(f"Помилка виділення ознак: {e}")
            return np.zeros(self._get_expected_feature_count())
    
    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Підготовка аудіо для аналізу"""
        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)))
        elif len(audio) > self.fft_size:
            start = (len(audio) - self.fft_size) // 2
            audio = audio[start:start + self.fft_size]
        
        return audio
    
    def _extract_fft_features(self, audio: np.ndarray) -> np.ndarray:
        """Виділення основних FFT ознак"""
        windowed_audio = audio * np.hanning(len(audio))
        
        fft = np.fft.fft(windowed_audio, n=self.fft_size)
        magnitude = np.abs(fft[:self.freq_bins])
        
        min_idx, max_idx = self.freq_range
        magnitude = magnitude[min_idx:max_idx]
        
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        magnitude_normalized = (magnitude_db - np.min(magnitude_db)) / \
                              (np.max(magnitude_db) - np.min(magnitude_db) + 1e-10)
        
        return magnitude_normalized
    
    def _extract_musical_features(self, audio: np.ndarray) -> np.ndarray:
        """Виділення музичних ознак"""
        features = []
        
        try:
            rms = np.sqrt(np.mean(audio ** 2))
            features.append(rms)
            
            zcr = np.mean(np.abs(np.diff(np.sign(audio))))
            features.append(zcr)
            
            fundamental_freq = self._estimate_fundamental_frequency(audio)
            features.append(fundamental_freq / self.max_freq)
            
        except Exception as e:
            logger.warning(f"Помилка виділення музичних ознак: {e}")
            features = [0.0, 0.0, 0.0]
        
        return np.array(features)
    
    def _extract_spectral_features(self, audio: np.ndarray) -> np.ndarray:
        """Виділення спектральних ознак"""
        features = []
        
        try:
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )[0]
            features.append(np.mean(spectral_centroid) / self.max_freq)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate
            )[0]
            features.append(np.mean(spectral_bandwidth) / self.max_freq)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate, roll_percent=0.85
            )[0]
            features.append(np.mean(spectral_rolloff) / self.max_freq)
            
        except Exception as e:
            logger.warning(f"Помилка виділення спектральних ознак: {e}")
            features = [0.0, 0.0, 0.0]
        
        return np.array(features)
    
    def _extract_harmonic_features(self, audio: np.ndarray) -> np.ndarray:
        """Виділення гармонічних ознак"""
        features = []
        
        try:
            fft = np.fft.fft(audio * np.hanning(len(audio)), n=self.fft_size)
            magnitude = np.abs(fft[:self.freq_bins])
            
            freqs = np.linspace(self.min_freq, self.max_freq, len(magnitude))
            
            peak_indices = self._find_spectral_peaks(magnitude)
            peak_features = self._calculate_peak_features(peak_indices, freqs, magnitude)
            features.extend(peak_features)
            
        except Exception as e:
            logger.warning(f"Помилка виділення гармонічних ознак: {e}")
            features = [0.0] * 15
        
        return np.array(features)
    
    def _estimate_fundamental_frequency(self, audio: np.ndarray) -> float:
        """Оцінка основної частоти"""
        try:
            pitches, magnitudes = librosa.piptrack(
                y=audio, sr=self.sample_rate, threshold=0.1
            )
            
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                return np.median(pitch_values)
            else:
                return 440.0
                
        except Exception:
            return 440.0
    
    def _find_spectral_peaks(self, magnitude: np.ndarray, prominence: float = 0.1):
        """Знаходження піків в спектрі"""
        try:
            peaks, _ = find_peaks(magnitude, prominence=prominence * np.max(magnitude))
            return peaks[:10].tolist()
        except:
            peaks = []
            for i in range(1, len(magnitude) - 1):
                if (magnitude[i] > magnitude[i-1] and 
                    magnitude[i] > magnitude[i+1] and 
                    magnitude[i] > prominence * np.max(magnitude)):
                    peaks.append(i)
                if len(peaks) >= 10:
                    break
            return peaks
    
    def _calculate_peak_features(self, peak_indices, freqs: np.ndarray, magnitude: np.ndarray):
        """Розрахунок ознак піків"""
        features = []
        
        if len(peak_indices) == 0:
            return [0.0] * 15
        
        fundamental_freq = freqs[peak_indices[0]] if peak_indices else 1.0
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
    
    def _get_expected_feature_count(self) -> int:
        """Розрахунок очікуваної кількості ознак"""
        min_idx, max_idx = self.freq_range
        fft_features = max_idx - min_idx
        musical_features = 3
        spectral_features = 3
        harmonic_features = 15
        
        return fft_features + musical_features + spectral_features + harmonic_features