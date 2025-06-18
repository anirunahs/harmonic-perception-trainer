import numpy as np
import librosa
from scipy.signal import butter, filtfilt
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    """Структура для зберігання аудіосегменту"""
    audio: np.ndarray
    start_time: float
    end_time: float
    confidence: float
    
@dataclass
class ProcessingResult:
    """Результат обробки аудіо"""
    segments: List[AudioSegment]
    original_duration: float
    processing_time: float
    quality_score: float
    warnings: List[str]

class RecognitionAudioProcessor:
    """Процесор для розпізнавання інтервалів в реальному часі"""
    
    def __init__(self, sr: int = 44100, segment_duration: float = 2.0):
        self.sr = sr
        self.segment_duration = segment_duration
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.high_pass_freq = 80
        self.low_pass_freq = 8000
        self.noise_gate_threshold = -60
        self.min_interval_gap = 0.5
        self.min_note_duration = 0.1
    
    def load_audio_from_blob(self, audio_data: bytes) -> np.ndarray:
        """Завантаження аудіо з бінарних даних"""
        import io
        try:
            audio_buffer = io.BytesIO(audio_data)
            audio, sr = librosa.load(audio_buffer, sr=self.sr, mono=True)
            self.logger.info(f"Завантажено з blob: {len(audio)/sr:.2f}с")
            return audio
        except Exception as e:
            self.logger.error(f"Помилка завантаження з blob: {e}")
            raise
    
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Видалення DC зміщення"""
        return audio - np.mean(audio)
    
    def apply_filters(self, audio: np.ndarray) -> np.ndarray:
        """Застосування high-pass та low-pass фільтрів"""
        nyquist = self.sr / 2
        
        if self.high_pass_freq < nyquist:
            high_normal = self.high_pass_freq / nyquist
            b_high, a_high = butter(4, high_normal, btype='high')
            audio = filtfilt(b_high, a_high, audio)
        
        if self.low_pass_freq < nyquist:
            low_normal = self.low_pass_freq / nyquist
            b_low, a_low = butter(4, low_normal, btype='low')
            audio = filtfilt(b_low, a_low, audio)
        
        return audio
    
    def noise_gate(self, audio: np.ndarray, threshold_db: float = -60) -> np.ndarray:
        """Спектральний гейт для зменшення шуму"""
        frame_length = 2048
        hop_length = 512
        
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        gate_mask = magnitude_db > threshold_db
        gated_magnitude = magnitude * gate_mask
        
        gated_stft = gated_magnitude * np.exp(1j * phase)
        return librosa.istft(gated_stft, hop_length=hop_length)
    
    def normalize_rms(self, audio: np.ndarray, target_rms_db: float = -18.0) -> np.ndarray:
        """RMS нормалізація"""
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms == 0:
            return audio
        
        current_rms_db = 20 * np.log10(current_rms)
        gain_db = target_rms_db - current_rms_db
        gain_linear = 10**(gain_db / 20)
        
        normalized = audio * gain_linear
        
        peak = np.max(np.abs(normalized))
        if peak > 0.95:
            normalized = normalized * (0.95 / peak)
        
        return normalized