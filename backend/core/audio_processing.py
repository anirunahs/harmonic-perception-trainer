import numpy as np
import librosa
from scipy.signal import butter, filtfilt, savgol_filter
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import io
import warnings
import json
import os
from django.conf import settings

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    """Структура для зберігання аудіосегменту"""
    audio: np.ndarray
    start_time: float
    end_time: float
    confidence: float
    energy: float = 0.0

@dataclass
class ProcessingResult:
    """Результат обробки аудіо"""
    segments: List[AudioSegment]
    original_duration: float
    processing_time: float
    quality_score: float
    warnings: List[str]
    signal_stats: dict
    calibration_info: dict = None

class DatasetCalibrator:
    """Калібратор для приведення до стандартів еталону"""
    
    def __init__(self):
        self.reference_params = self._load_or_create_reference_params()
        
    def _load_or_create_reference_params(self):
        """Завантаження еталонних параметрів"""
        params_file = os.path.join(settings.BASE_DIR, 'dataset_reference_simplified.json')
        
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                logger.info("Завантажено еталонні параметри")
                return params
            except Exception as e:
                logger.warning(f"Помилка завантаження параметрів: {e}")
        
        logger.info("Еталонні параметри за замовчуванням")
        return {
            'target_rms_db': -27.28,
            'rms_range': [-35.53, -20.52],
            'target_dynamic_range': 16.76,
            'target_spectral_centroid': 1095.32,
            'quality_thresholds': {
                'min_acceptable_rms_db': -37.78,
                'target_rms_db': -26.40,
                'max_acceptable_rms_db': -19.47,
                'ideal_rms_range': [-31.03, -22.78]
            }
        }
    
    def analyze_audio_compliance(self, audio: np.ndarray, sr: int = 44100) -> dict:
        """Аналіз відповідності аудіо стандартам датасету"""
        if len(audio) == 0:
            return {'compliant': False, 'issues': ['empty_audio']}
        
        # Базові параметри
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        if rms > 0:
            rms_db = 20 * np.log10(rms)
            peak_db = 20 * np.log10(peak)
            dynamic_range = peak_db - rms_db
        else:
            return {'compliant': False, 'issues': ['silent_audio']}
        
        # Перевірка відповідності
        target_rms = self.reference_params['target_rms_db']
        rms_range = self.reference_params['rms_range']
        
        issues = []
        
        # Перевірка RMS рівня
        if rms_db < rms_range[0]:
            issues.append('rms_too_low')
        elif rms_db > rms_range[1]:
            issues.append('rms_too_high')
        
        # Перевірка відхилення від цільового RMS
        rms_deviation = abs(rms_db - target_rms)
        if rms_deviation > 8.0:
            issues.append('rms_far_from_target')
        
        # Перевірка динамічного діапазону
        target_dr = self.reference_params['target_dynamic_range']
        if abs(dynamic_range - target_dr) > 12.0:
            issues.append('dynamic_range_mismatch')
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'current_rms_db': rms_db,
            'target_rms_db': target_rms,
            'rms_deviation': rms_deviation,
            'needs_calibration': len(issues) > 0,
            'quality_level': self._assess_quality_level(rms_db)
        }
    
    def _assess_quality_level(self, rms_db: float) -> str:
        """Оцінка рівня якості"""
        qt = self.reference_params['quality_thresholds']
        
        if qt['ideal_rms_range'][0] <= rms_db <= qt['ideal_rms_range'][1]:
            return 'ideal'
        elif qt['min_acceptable_rms_db'] <= rms_db <= qt['max_acceptable_rms_db']:
            return 'acceptable'
        elif rms_db < qt['min_acceptable_rms_db']:
            return 'too_quiet'
        else:
            return 'too_loud'
    
    def calibrate_to_dataset_standard(self, audio: np.ndarray, sr: int = 44100) -> tuple:
        """Калібрування аудіо"""
        calibrated_audio = audio.copy()
        calibration_steps = []
        
        # Видалення DC offset
        dc_offset = np.mean(calibrated_audio)
        if abs(dc_offset) > 0.001:
            calibrated_audio = calibrated_audio - dc_offset
            calibration_steps.append(f"DC offset: {dc_offset:.4f}")
        
        # Базова фільтрація
        try:
            calibrated_audio = self._highpass_filter(calibrated_audio, sr, 80)
            calibration_steps.append("High-pass 80Hz")
            
            calibrated_audio = self._lowpass_filter(calibrated_audio, sr, 8000)
            calibration_steps.append("Low-pass 8000Hz")
        except Exception as e:
            logger.warning(f"Помилка фільтрації: {e}")
        
        # RMS калібрування до цільового рівня
        current_rms = np.sqrt(np.mean(calibrated_audio**2))
        if current_rms > 0:
            current_rms_db = 20 * np.log10(current_rms)
            target_rms_db = self.reference_params['target_rms_db']
            
            gain_db = target_rms_db - current_rms_db
            gain_db = np.clip(gain_db, -20, 30)
            
            if abs(gain_db) > 2.0:
                gain_linear = 10**(gain_db / 20)
                calibrated_audio = calibrated_audio * gain_linear
                calibration_steps.append(f"RMS: {current_rms_db:.1f}→{target_rms_db:.1f}dB")
        
        # Контроль піків
        peak = np.max(np.abs(calibrated_audio))
        if peak > 0.95:
            safety_factor = 0.95 / peak
            calibrated_audio = calibrated_audio * safety_factor
            calibration_steps.append(f"Пік обмежено: {safety_factor:.3f}")
        
        # Tanh компресія
        calibrated_audio = np.tanh(calibrated_audio)
        calibration_steps.append("Tanh компресія")
        
        # Фінальна нормалізація
        peak = np.max(np.abs(calibrated_audio))
        if peak > 0:
            calibrated_audio = calibrated_audio / peak * 0.95
            calibration_steps.append("Фінальна нормалізація")
        
        final_analysis = self.analyze_audio_compliance(calibrated_audio, sr)
        
        calibration_info = {
            'steps_applied': calibration_steps,
            'final_compliance': final_analysis,
            'calibration_successful': final_analysis['compliant'] or final_analysis['quality_level'] in ['ideal', 'acceptable']
        }
        
        return calibrated_audio, calibration_info
    
    def _highpass_filter(self, audio: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
        """High-pass фільтр"""
        nyquist = sr / 2
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1.0:
            return audio
        b, a = butter(4, normal_cutoff, btype='high')
        return filtfilt(b, a, audio)
    
    def _lowpass_filter(self, audio: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
        """Low-pass фільтр"""
        nyquist = sr / 2
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1.0:
            return audio
        b, a = butter(4, normal_cutoff, btype='low')
        return filtfilt(b, a, audio)


class RecognitionAudioProcessor:
    """Процесор з калібруванням"""
    
    def __init__(self, sr: int = 44100, segment_duration: float = 2.0):
        self.sr = sr
        self.segment_duration = segment_duration
        self.target_samples = int(segment_duration * sr)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.calibrator = DatasetCalibrator()
        
        self.noise_gate_threshold = -55
        self.min_note_duration = 0.15
        self.gain_target_db = -18
    
    def load_audio_from_blob(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Завантаження аудіо"""
        try:
            audio_buffer = io.BytesIO(audio_data)
            
            audio, sr = librosa.load(
                audio_buffer, 
                sr=self.sr,
                mono=True,
                res_type='kaiser_fast'
            )
            
            self.logger.info(f"Завантажено: {len(audio)/sr:.2f}с, SR: {sr}")
            
            if len(audio) == 0:
                raise ValueError("Порожній аудіо масив")
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Помилка завантаження: {e}")
            raise
    
    def dataset_identical_preprocessing(self, audio: np.ndarray) -> tuple:
        """Обробка + автокалібрування"""
        
        compliance_analysis = self.calibrator.analyze_audio_compliance(audio, self.sr)
        
        self.logger.info(f"Аналіз відповідності: якість={compliance_analysis['quality_level']}, "
                        f"RMS={compliance_analysis['current_rms_db']:.1f}dB "
                        f"(ціль: {compliance_analysis['target_rms_db']:.1f}dB)")
        
        if compliance_analysis['needs_calibration']:
            self.logger.info("Застосовуємо автоматичне калібрування до стандартів датасету...")
            audio, calibration_info = self.calibrator.calibrate_to_dataset_standard(audio, self.sr)
            
            if calibration_info['calibration_successful']:
                self.logger.info("Калібрування успішне")
                for step in calibration_info['steps_applied']:
                    self.logger.info(f"  - {step}")
            else:
                self.logger.warning("Калібрування частково успішне")
        else:
            self.logger.info("Аудіо відповідає стандартам датасету")
            calibration_info = {'calibration_applied': False, 'reason': 'not_needed'}
        
        audio = audio - np.mean(audio)
        
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        
        audio = self._rms_normalize(audio, target_rms_db=self.gain_target_db)
        
        audio = np.tanh(audio)
        
        return audio, calibration_info
    
    def _rms_normalize(self, audio: np.ndarray, target_rms_db: float = -18) -> np.ndarray:
        """RMS нормалізація"""
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms == 0:
            return audio
        
        current_rms_db = 20 * np.log10(current_rms)
        gain_db = target_rms_db - current_rms_db
        
        gain_db = min(gain_db, 25)
        
        if abs(gain_db) > 1.0:
            gain_linear = 10**(gain_db / 20)
            normalized = audio * gain_linear
            
            peak = np.max(np.abs(normalized))
            if peak > 0.95:
                normalized = normalized * (0.95 / peak)
            
            return normalized
        
        return audio
    
    def dataset_identical_segmentation(self, audio: np.ndarray) -> List[AudioSegment]:
        """Сегментація"""
        segments = []
        
        if len(audio) >= self.target_samples:
            best_start = self._find_best_segment_start(audio, self.target_samples)
            best_audio = audio[best_start:best_start + self.target_samples]
            
            segment = AudioSegment(
                audio=best_audio,
                start_time=best_start / self.sr,
                end_time=(best_start + self.target_samples) / self.sr,
                confidence=self._calculate_segment_quality(best_audio),
                energy=np.mean(best_audio**2)
            )
            segments.append(segment)
            
        else:
            padded_audio = np.pad(audio, (0, self.target_samples - len(audio)), mode='constant')
            
            segment = AudioSegment(
                audio=padded_audio,
                start_time=0,
                end_time=len(padded_audio) / self.sr,
                confidence=self._calculate_segment_quality(padded_audio) * 0.9,
                energy=np.mean(padded_audio**2)
            )
            segments.append(segment)
        
        if len(audio) > self.target_samples * 1.5:
            end_start = len(audio) - self.target_samples
            end_audio = audio[end_start:end_start + self.target_samples]
            
            segment = AudioSegment(
                audio=end_audio,
                start_time=end_start / self.sr,
                end_time=len(audio) / self.sr,
                confidence=self._calculate_segment_quality(end_audio),
                energy=np.mean(end_audio**2)
            )
            segments.append(segment)
        
        segments.sort(key=lambda x: x.confidence, reverse=True)
        
        return segments[:3]
    
    def _find_best_segment_start(self, audio: np.ndarray, target_length: int) -> int:
        """Знаходження найкращого початку сегменту"""
        if len(audio) <= target_length:
            return 0
        
        best_start = 0
        best_energy = 0
        
        step = max(target_length // 10, 1)
        
        for start in range(0, len(audio) - target_length + 1, step):
            segment = audio[start:start + target_length]
            energy = np.mean(segment**2)
            
            if energy > best_energy:
                best_energy = energy
                best_start = start
        
        return best_start
    
    def _calculate_segment_quality(self, audio: np.ndarray) -> float:
        """Розрахунок якості сегменту"""
        if len(audio) == 0:
            return 0.0
        
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return 0.0
        
        clipping_ratio = np.sum(np.abs(audio) > 0.98) / len(audio)
        clipping_penalty = max(0, 1.0 - clipping_ratio * 10)
        
        peak = np.max(np.abs(audio))
        if peak > 0:
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
            dynamic_score = min(dynamic_range / 20, 1.0)
        else:
            dynamic_score = 0.0
        
        quality = min(rms * 10, 1.0) * clipping_penalty * dynamic_score
        
        return max(0.0, min(1.0, quality))
    
    def process(self, audio: np.ndarray) -> ProcessingResult:
        """Повна обробка з автоматичним калібруванням"""
        import time
        start_time = time.time()
        
        original_duration = len(audio) / self.sr
        warnings = []
        
        self.logger.info(f"Початок обробки з автокалібруванням: {original_duration:.2f}с")
        
        original_stats = self._analyze_signal_stats(audio)
        self.logger.info(f"Вхідний RMS: {original_stats.get('rms_db', 'N/A'):.1f}dB")
        
        try:
            processed_audio, calibration_info = self.dataset_identical_preprocessing(audio)
        except Exception as e:
            self.logger.error(f"Помилка калібрування/обробки: {e}")
            warnings.append(f"Помилка калібрування: {str(e)}")

            processed_audio = self._fallback_processing(audio)
            calibration_info = {'calibration_applied': False, 'error': str(e)}
        
        processed_stats = self._analyze_signal_stats(processed_audio)
        self.logger.info(f"Оброблений RMS: {processed_stats.get('rms_db', 'N/A'):.1f}dB")
        
        segments = self.dataset_identical_segmentation(processed_audio)
        
        if segments:
            quality_score = np.mean([seg.confidence for seg in segments])
        else:
            quality_score = 0.0
            warnings.append("Не вдалося створити сегменти")
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"Обробка завершена за {processing_time:.2f}с")
        self.logger.info(f"Сегментів: {len(segments)}, якість: {quality_score:.2f}")
        
        if calibration_info.get('calibration_applied', True):
            if calibration_info.get('calibration_successful', False):
                self.logger.info("Аудіо успішно калібровано")
            else:
                self.logger.warning("Калібрування частково успішне")
        
        return ProcessingResult(
            segments=segments,
            original_duration=original_duration,
            processing_time=processing_time,
            quality_score=quality_score,
            warnings=warnings,
            signal_stats={
                'original': original_stats,
                'processed': processed_stats,
                'improvement': processed_stats.get('rms_db', -100) - original_stats.get('rms_db', -100)
            },
            calibration_info=calibration_info
        )
    
    def _fallback_processing(self, audio: np.ndarray) -> np.ndarray:
        """Fallback обробка"""
        audio = audio - np.mean(audio)
        
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.8
        
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            current_rms_db = 20 * np.log10(rms)
            if current_rms_db < -30:
                gain_db = min(-20 - current_rms_db, 20)
                gain_linear = 10**(gain_db / 20)
                audio = audio * gain_linear
                
                peak = np.max(np.abs(audio))
                if peak > 0.95:
                    audio = audio * (0.95 / peak)
        
        return audio
    
    def _analyze_signal_stats(self, audio: np.ndarray) -> dict:
        """Аналіз статистики сигналу"""
        if len(audio) == 0:
            return {}
        
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        stats = {
            'rms': float(rms),
            'peak': float(peak),
            'rms_db': float(20 * np.log10(rms + 1e-10)),
            'peak_db': float(20 * np.log10(peak + 1e-10)),
            'duration': float(len(audio) / self.sr),
            'zero_crossings': int(np.sum(np.diff(np.sign(audio)) != 0)),
            'clipping_ratio': float(np.sum(np.abs(audio) > 0.99) / len(audio))
        }
        
        return stats