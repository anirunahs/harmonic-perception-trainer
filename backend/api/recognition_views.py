import os
import time
import numpy as np
import logging
import io
import base64
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework import serializers
import librosa
from scipy.signal import butter, filtfilt

from backend.core.audio_processing import RecognitionAudioProcessor
from backend.core.feature_extraction import FFTFeatureExtractor
from backend.core.model_inference import IntervalClassifier

logger = logging.getLogger(__name__)

class AudioRecognitionSerializer(serializers.Serializer):
    """Серіалізатор для запиту розпізнавання"""
    audio_data = serializers.CharField(help_text="Base64 encoded audio data")
    format = serializers.ChoiceField(
        choices=['wav', 'webm', 'mp3', 'ogg'],
        default='webm',
        help_text="Audio format"
    )
    preprocessing_level = serializers.ChoiceField(
        choices=['minimal', 'standard', 'aggressive'],
        default='standard',
        help_text="Level of audio preprocessing"
    )
    max_segments = serializers.IntegerField(
        default=3,
        min_value=1,
        max_value=5,
        help_text="Maximum number of segments to analyze"
    )

class IntervalRecognitionView(APIView):
    """Основний endpoint для розпізнавання музичних інтервалів"""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def __init__(self):
        super().__init__()
        try:
            self.processor = RecognitionAudioProcessor()
            self.feature_extractor = FFTFeatureExtractor()
            self.classifier = IntervalClassifier()
            self.is_ready = True
        except Exception as e:
            logger.error(f"Помилка ініціалізації моделей: {e}")
            self.is_ready = False
        
    def post(self, request):
        """Аналіз аудіо запису"""
        if not self.is_ready:
            return Response(
                {'error': 'Модель розпізнавання недоступна'}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
            
        serializer = AudioRecognitionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            result = self._process_audio_recognition(
                serializer.validated_data, 
                request.user.id
            )
            
            return Response({
                'status': 'completed',
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Помилка розпізнавання для користувача {request.user.id}: {e}")
            return Response(
                {'error': f'Помилка обробки: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _process_audio_recognition(self, validated_data, user_id):
        """Повна обробка аудіо для розпізнавання"""
        start_time = time.time()
        
        try:
            audio_b64 = validated_data['audio_data']
            audio_data = base64.b64decode(audio_b64)
            
            audio = self._load_and_preprocess_audio(audio_data, validated_data)
            
            if audio is None or len(audio) == 0:
                return {
                    'error': 'Не вдалося завантажити аудіо дані',
                    'status': 'error'
                }
            
            preprocessing_level = validated_data.get('preprocessing_level', 'standard')
            self._configure_preprocessing(preprocessing_level)
            
            processing_result = self.processor.process(audio)
            
            if not processing_result.segments:
                return {
                    'error': 'Не знайдено музичні сегменти в записі',
                    'warnings': processing_result.warnings,
                    'quality_score': processing_result.quality_score,
                    'status': 'error',
                    'processing_info': {
                        'duration': processing_result.original_duration,
                        'processing_time': processing_result.processing_time
                    }
                }
            
            max_segments = validated_data.get('max_segments', 3)
            segments_to_analyze = processing_result.segments[:max_segments]
            
            recognition_results = []
            
            for i, segment in enumerate(segments_to_analyze):
                try:
                    features = self.feature_extractor.extract_features(segment.audio)
                    prediction = self.classifier.predict(features)
                    
                    segment_result = {
                        'segment_id': i + 1,
                        'start_time': segment.start_time,
                        'end_time': segment.end_time,
                        'duration': segment.end_time - segment.start_time,
                        'confidence': segment.confidence,
                        'predictions': prediction['predictions'],
                        'best_prediction': prediction['best_prediction'],
                        'quality_metrics': {
                            'audio_quality': segment.confidence,
                            'prediction_confidence': prediction['best_prediction']['confidence'],
                            'overall_confidence': (segment.confidence + prediction['best_prediction']['confidence']) / 2
                        }
                    }
                    
                    recognition_results.append(segment_result)
                    
                except Exception as e:
                    logger.error(f"Помилка аналізу сегменту {i}: {e}")
                    continue
            
            total_processing_time = time.time() - start_time
            final_result = self._finalize_recognition_result(
                recognition_results, 
                processing_result,
                validated_data,
                total_processing_time
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Критична помилка обробки: {e}")
            return {
                'error': f'Критична помилка: {str(e)}',
                'status': 'error',
                'processing_info': {
                    'processing_time': time.time() - start_time
                }
            }
    
    def _load_and_preprocess_audio(self, audio_data, validated_data):
        """Завантаження та початкова обробка аудіо"""
        try:
            audio_buffer = io.BytesIO(audio_data)
            
            audio, sr = librosa.load(
                audio_buffer,
                sr=44100,
                mono=True,
                offset=0.0,
                duration=None
            )
            
            logger.info(f"Завантажено аудіо: {len(audio)/sr:.2f}с, SR: {sr}")
            
            audio = self._clean_audio(audio, sr)
            
            audio = self._trim_silence(audio, sr)
            
            min_duration = 1.0
            if len(audio) / sr < min_duration:
                logger.warning(f"Аудіо занадто коротке: {len(audio)/sr:.2f}с")
                return None
            
            return audio
            
        except Exception as e:
            logger.error(f"Помилка завантаження аудіо: {e}")
            return None
    
    def _clean_audio(self, audio, sr):
        """Очищення аудіо від артефактів"""
        audio = audio - np.mean(audio)
        
        try:
            nyquist = sr / 2
            lowcut = 80  # Гц
            high_normal = lowcut / nyquist
            
            if high_normal < 1.0:
                b, a = butter(4, high_normal, btype='high')
                audio = filtfilt(b, a, audio)
        except Exception as e:
            logger.warning(f"Помилка high-pass фільтра: {e}")
        
        try:
            highcut = 8000  # Гц
            low_normal = highcut / nyquist
            
            if low_normal < 1.0:
                b, a = butter(4, low_normal, btype='low')
                audio = filtfilt(b, a, audio)
        except Exception as e:
            logger.warning(f"Помилка low-pass фільтра: {e}")
        
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.8
        
        return audio
    
    def _trim_silence(self, audio, sr, top_db=25):
        """Видалення тиші на початку та в кінці"""
        try:
            intervals = librosa.effects.split(audio, top_db=top_db)
            
            if len(intervals) > 0:
                start_sample = intervals[0][0]
                end_sample = intervals[-1][1]
                
                buffer_samples = int(0.1 * sr)
                start_sample = max(0, start_sample - buffer_samples)
                end_sample = min(len(audio), end_sample + buffer_samples)
                
                audio = audio[start_sample:end_sample]
                logger.info(f"Обрізано тишу: {start_sample/sr:.2f}с - {end_sample/sr:.2f}с")
            
            return audio
            
        except Exception as e:
            logger.warning(f"Помилка видалення тиші: {e}")
            return audio
    
    def _configure_preprocessing(self, level):
        """Налаштування рівня попередньої обробки"""
        if level == 'aggressive':
            self.processor.noise_gate_threshold = -50
            self.processor.min_note_duration = 0.05
        elif level == 'minimal':
            self.processor.noise_gate_threshold = -70
            self.processor.min_note_duration = 0.2
        else:
            self.processor.noise_gate_threshold = -60
            self.processor.min_note_duration = 0.1
    
    def _finalize_recognition_result(self, recognition_results, processing_result, validated_data, total_time):
        """Формування підсумкового результату"""
        
        if not recognition_results:
            return {
                'status': 'error',
                'message': 'Не вдалося розпізнати інтервали',
                'processing_info': {
                    'duration': processing_result.original_duration,
                    'processing_time': total_time,
                    'quality_score': processing_result.quality_score,
                    'warnings': processing_result.warnings
                }
            }
        
        best_result = max(recognition_results, 
                         key=lambda x: x['quality_metrics']['overall_confidence'])
        
        all_predictions = []
        for result in recognition_results:
            all_predictions.extend(result['predictions'])
        
        interval_stats = {}
        for pred in all_predictions:
            interval = pred['interval']
            if interval not in interval_stats:
                interval_stats[interval] = {
                    'count': 0,
                    'total_confidence': 0,
                    'avg_confidence': 0
                }
            interval_stats[interval]['count'] += 1
            interval_stats[interval]['total_confidence'] += pred['confidence']
        
        for interval, stats in interval_stats.items():
            stats['avg_confidence'] = stats['total_confidence'] / stats['count']
        
        top_intervals = sorted(
            interval_stats.items(),
            key=lambda x: (x[1]['count'], x[1]['avg_confidence']),
            reverse=True
        )[:3]
        
        return {
            'status': 'success',
            'best_prediction': {
                'interval': best_result['best_prediction']['interval'],
                'confidence': best_result['quality_metrics']['overall_confidence'],
                'segment_info': {
                    'start_time': best_result['start_time'],
                    'end_time': best_result['end_time'],
                    'audio_quality': best_result['quality_metrics']['audio_quality']
                }
            },
            'alternative_predictions': [
                {
                    'interval': interval,
                    'confidence': stats['avg_confidence'],
                    'occurrence_count': stats['count']
                }
                for interval, stats in top_intervals[1:]
            ],
            'segments_analyzed': len(recognition_results),
            'detailed_results': recognition_results,
            'processing_info': {
                'original_duration': processing_result.original_duration,
                'processing_time': total_time,
                'quality_score': processing_result.quality_score,
                'warnings': processing_result.warnings,
                'preprocessing_level': validated_data.get('preprocessing_level', 'standard')
            },
            'recommendations': self._generate_recommendations(recognition_results, processing_result)
        }
    
    def _generate_recommendations(self, recognition_results, processing_result):
        """Генерація рекомендацій для користувача"""
        recommendations = []
        
        if not recognition_results:
            return recommendations
        
        avg_audio_quality = np.mean([r['quality_metrics']['audio_quality'] 
                                   for r in recognition_results])
        avg_prediction_confidence = np.mean([r['quality_metrics']['prediction_confidence'] 
                                           for r in recognition_results])
        
        if avg_audio_quality < 0.5:
            recommendations.append({
                'type': 'audio_quality',
                'message': 'Низька якість аудіо. Спробуйте записати в тишому місці з кращим мікрофоном.',
                'severity': 'warning'
            })
        
        if avg_prediction_confidence < 0.6:
            recommendations.append({
                'type': 'recognition',
                'message': 'Низька впевненість розпізнавання. Переконайтеся, що граєте чіткі музичні інтервали.',
                'severity': 'info'
            })
        
        if processing_result.quality_score < 0.4:
            recommendations.append({
                'type': 'recording',
                'message': 'Спробуйте записати інтервали повільніше та чіткіше. Уникайте фонових шумів.',
                'severity': 'warning'
            })
        
        if len(recognition_results) == 1 and recognition_results[0]['duration'] < 1.5:
            recommendations.append({
                'type': 'duration',
                'message': 'Зробіть довший запис (2-3 секунди) для кращого аналізу.',
                'severity': 'info'
            })
        
        if processing_result.original_duration < 2.0:
            recommendations.append({
                'type': 'short_recording',
                'message': 'Короткий запис може знизити точність. Рекомендована тривалість: 2-3 секунди.',
                'severity': 'info'
            })
        
        low_quality_segments = [r for r in recognition_results 
                               if r['quality_metrics']['overall_confidence'] < 0.5]
        
        if len(low_quality_segments) > len(recognition_results) * 0.7:
            recommendations.append({
                'type': 'segment_quality',
                'message': 'Багато сегментів низької якості. Спробуйте грати інтервали більш виразно.',
                'severity': 'warning'
            })
        
        return recommendations


class QuickRecognitionView(APIView):
    """Швидке розпізнавання з мінімальною обробкою"""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def __init__(self):
        super().__init__()
        try:
            self.feature_extractor = FFTFeatureExtractor()
            self.classifier = IntervalClassifier()
            self.is_ready = True
        except Exception as e:
            logger.error(f"Помилка ініціалізації для швидкого розпізнавання: {e}")
            self.is_ready = False
    
    def post(self, request):
        """Швидкий аналіз (спрощена обробка)"""
        if not self.is_ready:
            return Response(
                {'error': 'Модель розпізнавання недоступна'}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
            
        serializer = AudioRecognitionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            start_time = time.time()
            
            audio_b64 = serializer.validated_data['audio_data']
            audio_data = base64.b64decode(audio_b64)
            
            audio_buffer = io.BytesIO(audio_data)
            audio, sr = librosa.load(audio_buffer, sr=22050, mono=True)
            
            if len(audio) == 0:
                return Response(
                    {'error': 'Порожній аудіо файл'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            audio = self._minimal_processing(audio, sr)
            
            features = self.feature_extractor.extract_features(audio)
            
            prediction = self.classifier.predict(features)
            
            processing_time = time.time() - start_time
            
            return Response({
                'status': 'success',
                'mode': 'quick',
                'prediction': prediction['best_prediction'],
                'alternatives': prediction['predictions'][:3],
                'processing_time': processing_time,
                'note': 'Швидкий режим - спрощена обробка'
            })
            
        except Exception as e:
            logger.error(f"Помилка швидкого розпізнавання: {e}")
            return Response(
                {'error': f'Помилка обробки: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _minimal_processing(self, audio, sr):
        """Мінімальна обробка для швидкого режиму"""
        audio = audio - np.mean(audio)
        
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 0.1
            audio = audio * (target_rms / rms)
        
        target_samples = int(2.0 * sr)
        if len(audio) >= target_samples:
            start = (len(audio) - target_samples) // 2
            audio = audio[start:start + target_samples]
        else:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        
        return audio


class ModelStatusView(APIView):
    """Перевірка статусу моделі розпізнавання"""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def get(self, request):
        """Отримання статусу моделі"""
        try:
            classifier = IntervalClassifier()
            model_info = classifier.get_model_info()
            
            return Response({
                'status': 'ready' if model_info.get('loaded', False) else 'error',
                'model_info': model_info,
                'available_endpoints': [
                    '/api/recognition/interval/',
                    '/api/recognition/quick/',
                    '/api/recognition/status/'
                ]
            })
            
        except Exception as e:
            logger.error(f"Помилка перевірки статусу моделі: {e}")
            return Response({
                'status': 'error',
                'error': str(e),
                'model_info': {'loaded': False}
            })


class AudioDiagnosticsView(APIView):
    """Діагностика аудіо без розпізнавання"""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def post(self, request):
        """Аналіз якості аудіо без розпізнавання"""
        try:
            audio_data = base64.b64decode(request.data.get('audio_data', ''))
            
            if not audio_data:
                return Response(
                    {'error': 'Відсутні аудіо дані'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            audio_buffer = io.BytesIO(audio_data)
            audio, sr = librosa.load(audio_buffer, sr=44100, mono=True)
            
            diagnostics = self._analyze_audio_quality(audio, sr)
            
            return Response({
                'status': 'success',
                'diagnostics': diagnostics
            })
            
        except Exception as e:
            logger.error(f"Помилка діагностики аудіо: {e}")
            return Response(
                {'error': f'Помилка аналізу: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _analyze_audio_quality(self, audio, sr):
        """Детальний аналіз якості аудіо"""
        duration = len(audio) / sr
        
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
        
        rms_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        dynamic_range = peak_db - rms_db
        
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            mean_centroid = np.mean(spectral_centroid)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            mean_bandwidth = np.mean(spectral_bandwidth)
        except:
            mean_centroid = 0
            mean_bandwidth = 0
        
        silence_threshold = peak * 0.01
        silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
        
        quality_score = self._calculate_quality_score(
            rms_db, clipping_ratio, dynamic_range, silence_ratio
        )
        
        return {
            'duration': round(duration, 2),
            'rms_db': round(rms_db, 2),
            'peak_db': round(peak_db, 2),
            'dynamic_range': round(dynamic_range, 2),
            'clipping_ratio': round(clipping_ratio * 100, 2),
            'silence_ratio': round(silence_ratio * 100, 2),
            'spectral_centroid': round(mean_centroid, 2),
            'spectral_bandwidth': round(mean_bandwidth, 2),
            'quality_score': round(quality_score, 2),
            'quality_rating': self._get_quality_rating(quality_score),
            'recommendations': self._get_audio_recommendations(
                rms_db, clipping_ratio, dynamic_range, silence_ratio, duration
            )
        }
    
    def _calculate_quality_score(self, rms_db, clipping_ratio, dynamic_range, silence_ratio):
        """Розрахунок загального скору якості"""
        score = 100
        
        # Штраф за низький рівень
        if rms_db < -40:
            score -= (40 + rms_db)
        elif rms_db < -20:
            score -= (20 + rms_db) * 0.5
        
        # Штраф за кліпування
        score -= clipping_ratio * 1000
        
        # Штраф за низький динамічний діапазон
        if dynamic_range < 10:
            score -= (10 - dynamic_range) * 2
        
        # Штраф за надмірну тишу
        if silence_ratio > 0.5:
            score -= (silence_ratio - 0.5) * 100
        
        return max(0, min(100, score))
    
    def _get_quality_rating(self, score):
        """Текстова оцінка якості"""
        if score >= 80:
            return 'Відмінна'
        elif score >= 60:
            return 'Хороша'
        elif score >= 40:
            return 'Задовільна'
        elif score >= 20:
            return 'Погана'
        else:
            return 'Дуже погана'
    
    def _get_audio_recommendations(self, rms_db, clipping_ratio, dynamic_range, silence_ratio, duration):
        """Рекомендації по покращенню якості"""
        recommendations = []
        
        if rms_db < -40:
            recommendations.append("Збільшіть гучність запису або приблизьтеся до мікрофона")
        
        if clipping_ratio > 0.01:
            recommendations.append("Зменшіть гучність для уникнення кліпування")
        
        if dynamic_range < 10:
            recommendations.append("Грайте з більшою динамікою для кращого розпізнавання")
        
        if silence_ratio > 0.7:
            recommendations.append("Зменшіть паузи між нотами або грайте голосніше")
        
        if duration < 2:
            recommendations.append("Зробіть довший запис (2-3 секунди) для кращого аналізу")
        
        if not recommendations:
            recommendations.append("Якість запису хороша для розпізнавання")
        
        return recommendations