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
    
    def extract_music_segments(self, audio: np.ndarray) -> List[AudioSegment]:
        """Виділення музичних сегментів з аудіо"""
        segments = []
        
        intervals = librosa.effects.split(audio, top_db=25, hop_length=512)
        
        for start_frame, end_frame in intervals:
            start_time = start_frame / self.sr
            end_time = end_frame / self.sr
            duration = end_time - start_time
            
            if duration >= self.min_note_duration:
                segment_audio = audio[start_frame:end_frame]
                
                quality_score = self.analyze_segment_quality(segment_audio)
                
                if quality_score > 0.3:
                    segment = AudioSegment(
                        audio=segment_audio,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=quality_score
                    )
                    segments.append(segment)
        
        return segments
    
    def analyze_segment_quality(self, audio: np.ndarray) -> float:
        """Аналіз якості аудіосегменту"""
        if len(audio) == 0:
            return 0.0
        
        metrics = []
        
        rms = np.sqrt(np.mean(audio**2))
        if rms > 1e-6:
            snr_score = min(rms * 100, 1.0)
            metrics.append(snr_score)
        
        try:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
            centroid_score = 1.0 - abs(np.mean(centroid) - 2000) / 4000
            metrics.append(max(0.0, centroid_score))
        except:
            metrics.append(0.5)
        
        peak = np.max(np.abs(audio))
        if peak > 0 and rms > 0:
            dynamic_range = peak / rms
            dr_score = min(dynamic_range / 10, 1.0)
            metrics.append(dr_score)
        
        return np.mean(metrics) if metrics else 0.0
    
    def create_optimal_segments(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """Створення оптимальних сегментів для розпізнавання"""
        optimal_segments = []
        target_samples = int(self.segment_duration * self.sr)
        
        for segment in segments:
            segment_length = len(segment.audio)
            
            if segment_length >= target_samples:
                best_start = self.find_best_subsegment(segment.audio, target_samples)
                optimal_audio = segment.audio[best_start:best_start + target_samples]
                
                optimal_segment = AudioSegment(
                    audio=optimal_audio,
                    start_time=segment.start_time + best_start/self.sr,
                    end_time=segment.start_time + (best_start + target_samples)/self.sr,
                    confidence=segment.confidence
                )
                optimal_segments.append(optimal_segment)
                
            elif segment_length >= target_samples * 0.7:
                padding_needed = target_samples - segment_length
                padded_audio = np.pad(segment.audio, (0, padding_needed), mode='constant')
                
                optimal_segment = AudioSegment(
                    audio=padded_audio,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence * 0.9
                )
                optimal_segments.append(optimal_segment)
        
        optimal_segments.sort(key=lambda x: x.confidence, reverse=True)
        
        return optimal_segments[:3]
    
    def find_best_subsegment(self, audio: np.ndarray, target_length: int) -> int:
        """Знаходження найкращого підсегменту заданої довжини"""
        if len(audio) <= target_length:
            return 0
        
        best_start = 0
        best_score = 0
        
        step = max(target_length // 10, 1)
        
        for start in range(0, len(audio) - target_length + 1, step):
            subsegment = audio[start:start + target_length]
            score = self.analyze_segment_quality(subsegment)
            
            if score > best_score:
                best_score = score
                best_start = start
        
        return best_start
    
    def process(self, audio: np.ndarray) -> ProcessingResult:
        """Повна обробка аудіо для розпізнавання"""
        import time
        start_time = time.time()
        
        original_duration = len(audio) / self.sr
        warnings = []
        
        self.logger.info(f"Початок обробки аудіо: {original_duration:.2f}с")
        
        audio = self.remove_dc_offset(audio)
        audio = self.apply_filters(audio)
        
        try:
            audio = self.noise_gate(audio, self.noise_gate_threshold)
        except Exception as e:
            self.logger.warning(f"Помилка шумоподавлення: {e}")
            warnings.append("Шумоподавлення не застосовано")
        
        try:
            audio = self.normalize_rms(audio)
        except Exception as e:
            self.logger.warning(f"Помилка нормалізації: {e}")
            warnings.append("Помилка нормалізації")
        
        raw_segments = self.extract_music_segments(audio)
        self.logger.info(f"Знайдено {len(raw_segments)} сирих сегментів")
        
        if not raw_segments:
            warnings.append("Не знайдено музичних сегментів")
            target_samples = int(self.segment_duration * self.sr)
            if len(audio) >= target_samples:
                best_start = self.find_best_subsegment(audio, target_samples)
                segment_audio = audio[best_start:best_start + target_samples]
            else:
                segment_audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
            
            fallback_segment = AudioSegment(
                audio=segment_audio,
                start_time=0,
                end_time=len(segment_audio) / self.sr,
                confidence=0.5
            )
            raw_segments = [fallback_segment]
        
        optimal_segments = self.create_optimal_segments(raw_segments)
        
        if optimal_segments:
            quality_score = np.mean([seg.confidence for seg in optimal_segments])
        else:
            quality_score = 0.0
        
        processing_time = time.time() - start_time
        self.logger.info(f"Обробка завершена за {processing_time:.2f}с, якість: {quality_score:.2f}")
        
        return ProcessingResult(
            segments=optimal_segments,
            original_duration=original_duration,
            processing_time=processing_time,
            quality_score=quality_score,
            warnings=warnings
        )