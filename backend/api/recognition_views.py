import os
import time
import numpy as np
import logging
import io
import base64
from rest_framework import status, serializers
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
import librosa

from core.audio_processing import RecognitionAudioProcessor
from core.feature_extraction import FFTFeatureExtractor  
from core.model_inference import IntervalClassifier

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
        self.processor = None
        self.feature_extractor = None
        self.classifier = None
        self.is_ready = False
        
        try:
            self.processor = RecognitionAudioProcessor()
            self.feature_extractor = FFTFeatureExtractor()
            self.classifier = IntervalClassifier()
            
            if self.classifier.is_loaded:
                self.is_ready = True
                logger.info("Система розпізнавання ініціалізована успішно")
            else:
                logger.error("Модель не завантажена")
                
        except Exception as e:
            logger.error(f"Помилка ініціалізації: {e}")
            self.is_ready = False
        
    def post(self, request):
        """Аналіз аудіо"""
        if not self.is_ready:
            return Response(
                {
                    'error': 'Модель розпізнавання недоступна',
                    'details': 'Система не змогла ініціалізувати модель',
                    'status': 'service_unavailable'
                }, 
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
            
            user_message = self._get_user_friendly_error(e)
            
            return Response(
                {
                    'error': user_message,
                    'status': 'error',
                    'technical_details': str(e) if request.user.is_staff else None
                }, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_user_friendly_error(self, exception):
        """Перетворення технічних помилок в зрозумілі"""
        error_str = str(exception).lower()
        
        if 'format not recognised' in error_str or 'could not open' in error_str:
            return 'Не вдалося розпізнати формат аудіо. Спробуйте записати ще раз.'
        elif 'tensorflow' in error_str or 'model' in error_str:
            return 'Помилка моделі розпізнавання. Спробуйте пізніше.'
        elif 'memory' in error_str:
            return 'Недостатньо пам\'яті. Спробуйте зробити коротший запис.'
        elif 'empty' in error_str or 'no data' in error_str:
            return 'Порожній аудіо файл. Переконайтеся, що запис зроблено правильно.'
        elif 'inputlayer' in error_str or 'batch_shape' in error_str:
            return 'Проблема сумісності моделі.'
        else:
            return 'Помилка обробки аудіо. Спробуйте ще раз.'
    
    def _process_audio_recognition(self, validated_data, user_id):
        """Обробка аудіо для розпізнавання"""
        start_time = time.time()
        
        try:
            audio_b64 = validated_data['audio_data']
            try:
                audio_data = base64.b64decode(audio_b64)
            except Exception as e:
                return {
                    'error': 'Помилка декодування аудіо даних',
                    'status': 'error'
                }
            
            try:
                audio = self.processor.load_audio_from_blob(audio_data)
                if audio is None or len(audio) == 0:
                    raise ValueError("Порожній аудіо файл")
            except Exception as e:
                logger.error(f"Помилка завантаження аудіо: {e}")
                return {
                    'error': 'Не вдалося завантажити аудіо',
                    'status': 'error'
                }
            
            try:
                processing_result = self.processor.process(audio)
            except Exception as e:
                logger.error(f"Помилка обробки аудіо: {e}")
                return {
                    'error': 'Помилка обробки аудіо сигналу',
                    'status': 'error'
                }
            
            if not processing_result.segments:
                return {
                    'error': 'Не знайдено музичні сегменти в записі',
                    'status': 'error',
                    'processing_info': {
                        'duration': processing_result.original_duration,
                        'quality_score': processing_result.quality_score,
                        'warnings': processing_result.warnings
                    }
                }
            
            max_segments = validated_data.get('max_segments', 3)
            segments_to_analyze = processing_result.segments[:max_segments]
            
            recognition_results = []
            
            for i, segment in enumerate(segments_to_analyze):
                try:
                    features = self.feature_extractor.extract_features(segment.audio)
                    
                    prediction = self.classifier.predict(features)
                    
                    if prediction.get('error', False):
                        logger.warning(f"Помилка предикції для сегменту {i}: {prediction.get('message', 'unknown')}")
                        continue
                    
                    top_predictions = prediction['predictions'][:3] if 'predictions' in prediction else []
                    
                    segment_result = {
                        'segment_id': i + 1,
                        'start_time': segment.start_time,
                        'end_time': segment.end_time,
                        'duration': segment.end_time - segment.start_time,
                        'confidence': segment.confidence,
                        'top_predictions': top_predictions,
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
            
            if not recognition_results:
                return {
                    'error': 'Не вдалося проаналізувати жоден сегмент',
                    'status': 'error',
                    'processing_info': {
                        'segments_found': len(processing_result.segments),
                        'segments_analyzed': 0
                    }
                }
            
            total_processing_time = time.time() - start_time
            final_result = self._finalize_recognition_result(
                recognition_results, 
                processing_result,
                validated_data,
                total_processing_time
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Критична помилка: {e}")
            return {
                'error': 'Критична помилка системи',
                'status': 'error',
                'technical_details': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _finalize_recognition_result(self, recognition_results, processing_result, validated_data, total_time):
        """Формування підсумкового результату з детальними даними по сегментах"""
        
        best_result = max(recognition_results, 
                         key=lambda x: x['quality_metrics']['overall_confidence'])
        
        all_predictions = []
        for result in recognition_results:
            all_predictions.extend(result['top_predictions'])
        
        interval_stats = {}
        for pred in all_predictions:
            interval = pred['interval']
            if interval not in interval_stats:
                interval_stats[interval] = {'count': 0, 'total_confidence': 0}
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
            'segments_analysis': [
                {
                    'segment_id': result['segment_id'],
                    'time_range': f"{result['start_time']:.1f}с - {result['end_time']:.1f}с",
                    'duration': f"{result['duration']:.1f}с",
                    'audio_quality': result['quality_metrics']['audio_quality'],
                    'top_predictions': result['top_predictions'],
                    'best_interval': result['best_prediction']['interval'],
                    'best_confidence': result['best_prediction']['confidence']
                }
                for result in recognition_results
            ],
            'overall_top_intervals': [
                {
                    'interval': interval,
                    'confidence': stats['avg_confidence'],
                    'occurrence_count': stats['count'],
                    'segments_found_in': stats['count']
                }
                for interval, stats in top_intervals
            ],
            'segments_analyzed': len(recognition_results),
            'processing_info': {
                'original_duration': processing_result.original_duration,
                'processing_time': total_time,
                'quality_score': processing_result.quality_score,
                'warnings': processing_result.warnings,
                'preprocessing_level': validated_data.get('preprocessing_level', 'standard'),
                'segments_found': len(processing_result.segments),
                'segments_used': len(recognition_results),
                'signal_improvement': processing_result.signal_stats.get('improvement', 0)
            }
        }


class QuickRecognitionView(APIView):
    """Швидке розпізнавання зі спрощеною обробкою"""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = None
        self.classifier = None
        self.is_ready = False
        
        try:
            self.feature_extractor = FFTFeatureExtractor()
            self.classifier = IntervalClassifier()
            
            if self.classifier.is_loaded:
                self.is_ready = True
                logger.info("Швидке розпізнавання ініціалізовано")
            else:
                logger.error("Модель не завантажена для швидкого розпізнавання")
                
        except Exception as e:
            logger.error(f"Помилка ініціалізації швидкого розпізнавання: {e}")
            self.is_ready = False
    
    def post(self, request):
        """Швидкий аналіз з мінімальною обробкою"""
        if not self.is_ready:
            return Response(
                {
                    'error': 'Модель розпізнавання недоступна для швидкого аналізу',
                    'status': 'service_unavailable'
                }, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
            
        serializer = AudioRecognitionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            start_time = time.time()
            
            audio_b64 = serializer.validated_data['audio_data']
            try:
                audio_data = base64.b64decode(audio_b64)
                audio_buffer = io.BytesIO(audio_data)
                audio, sr = librosa.load(audio_buffer, sr=22050, mono=True)
            except Exception as e:
                return Response(
                    {'error': 'Не вдалося завантажити аудіо файл'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if len(audio) == 0:
                return Response(
                    {'error': 'Порожній аудіо файл'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                audio = self._quick_processing_dataset_style(audio, sr)
            except Exception as e:
                logger.error(f"Помилка швидкої обробки: {e}")
                return Response(
                    {'error': 'Помилка обробки аудіо'}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            try:
                features = self.feature_extractor.extract_features(audio)
                prediction = self.classifier.predict(features)
                
                if prediction.get('error', False):
                    raise Exception(prediction.get('message', 'Помилка розпізнавання'))
                
            except Exception as e:
                logger.error(f"Помилка швидкого розпізнавання: {e}")
                return Response(
                    {'error': 'Помилка розпізнавання інтервалу'}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            processing_time = time.time() - start_time
            
            return Response({
                'status': 'success',
                'mode': 'quick',
                'prediction': prediction['best_prediction'],
                'top_predictions': prediction['predictions'][:3],
                'processing_time': processing_time,
                'note': 'Швидкий режим з базовою обробкою'
            })
            
        except Exception as e:
            logger.error(f"Критична помилка швидкого розпізнавання: {e}")
            return Response(
                {'error': 'Системна помилка швидкого розпізнавання'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _quick_processing_dataset_style(self, audio, sr):
        """Швидка обробка з елементами логіки датасету"""
        audio = audio - np.mean(audio)
        
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            current_rms_db = 20 * np.log10(rms)
            target_rms_db = -18
            gain_db = min(target_rms_db - current_rms_db, 20)
            
            if gain_db > 0:
                gain_linear = 10**(gain_db / 20)
                audio = audio * gain_linear
                
                peak = np.max(np.abs(audio))
                if peak > 0.95:
                    audio = audio * (0.90 / peak)
        
        audio = np.tanh(audio)
        
        target_samples = int(2.0 * sr)
        if len(audio) >= target_samples:
            best_start = 0
            best_energy = 0
            step = max(target_samples // 10, 1)
            
            for start in range(0, len(audio) - target_samples + 1, step):
                segment = audio[start:start + target_samples]
                energy = np.mean(segment**2)
                if energy > best_energy:
                    best_energy = energy
                    best_start = start
                    
            audio = audio[best_start:best_start + target_samples]
        else:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        
        return audio


class ModelStatusView(APIView):
    """Перевірка статусу моделі"""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def get(self, request):
        """Отримання статусу системи"""
        try:
            classifier = IntervalClassifier()
            model_info = classifier.get_model_info()
            
            try:
                processor = RecognitionAudioProcessor()
                processor_status = True
            except Exception as e:
                processor_status = False
                logger.error(f"Помилка ініціалізації процесора: {e}")
            
            try:
                extractor = FFTFeatureExtractor()
                extractor_status = True
            except Exception as e:
                extractor_status = False
                logger.error(f"Помилка ініціалізації екстрактора: {e}")
            
            overall_status = (
                model_info.get('loaded', False) and 
                processor_status and 
                extractor_status
            )
            
            return Response({
                'status': 'ready' if overall_status else 'error',
                'components': {
                    'model': {
                        'status': 'ready' if model_info.get('loaded', False) else 'error',
                        'info': model_info
                    },
                    'audio_processor': {
                        'status': 'ready' if processor_status else 'error'
                    },
                    'feature_extractor': {
                        'status': 'ready' if extractor_status else 'error'
                    }
                },
                'system_info': {
                    'tensorflow_version': model_info.get('tensorflow_version', 'unknown'),
                    'processor_version': 'dataset_identical_v1',
                    'supported_intervals': [
                        'major_2nd', 'major_3rd', 'major_6th', 'major_7th', 
                        'minor_2nd', 'minor_3rd', 'minor_6th', 'minor_7th', 
                        'perfect_4th', 'perfect_5th', 'perfect_8th', 'tritone'
                    ]
                },
                'available_endpoints': [
                    '/api/recognition/interval/',
                    '/api/recognition/quick/',
                    '/api/recognition/status/',
                    '/api/recognition/diagnostics/'
                ]
            })
            
        except Exception as e:
            logger.error(f"Помилка перевірки статусу: {e}")
            return Response({
                'status': 'error',
                'error': str(e),
                'components': {
                    'model': {'status': 'error'},
                    'audio_processor': {'status': 'unknown'},
                    'feature_extractor': {'status': 'unknown'}
                }
            })


class AudioDiagnosticsView(APIView):
    """Базова діагностика аудіо"""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def post(self, request):
        """Аналіз якості аудіо"""
        try:
            audio_data = base64.b64decode(request.data.get('audio_data', ''))
            
            if not audio_data:
                return Response(
                    {'error': 'Відсутні аудіо дані'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                audio_buffer = io.BytesIO(audio_data)
                audio, sr = librosa.load(audio_buffer, sr=44100, mono=True)
            except Exception as e:
                return Response(
                    {'error': 'Не вдалося завантажити аудіо для діагностики'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            duration = len(audio) / sr
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            
            rms_db = 20 * np.log10(rms + 1e-10)
            peak_db = 20 * np.log10(peak + 1e-10)
            
            if rms_db > -25:
                quality = 'good'
            elif rms_db > -40:
                quality = 'fair'
            else:
                quality = 'poor'
            
            diagnostics = {
                'duration': round(duration, 2),
                'rms_db': round(rms_db, 1),
                'peak_db': round(peak_db, 1),
                'quality': quality,
                'recommendations': self._generate_basic_recommendations(rms_db, duration, peak)
            }
            
            return Response({
                'status': 'success',
                'diagnostics': diagnostics
            })
            
        except Exception as e:
            logger.error(f"Помилка діагностики: {e}")
            return Response(
                {'error': 'Помилка аналізу аудіо'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _generate_basic_recommendations(self, rms_db, duration, peak):
        """Базові рекомендації"""
        recommendations = []
        
        if rms_db < -40:
            recommendations.append({
                'category': 'volume',
                'message': f'Тихий запис ({rms_db:.1f}dB). Записуйте ближче до мікрофона.',
                'priority': 'high'
            })
        
        if duration < 1.5:
            recommendations.append({
                'category': 'duration',
                'message': 'Короткий запис. Рекомендована тривалість: 2-3 секунди.',
                'priority': 'medium'
            })
        
        if peak > 0.99:
            recommendations.append({
                'category': 'clipping',
                'message': 'Виявлено кліпування. Зменшіть гучність запису.',
                'priority': 'high'
            })
        
        if not recommendations:
            recommendations.append({
                'category': 'quality',
                'message': 'Хороша якість запису для розпізнавання.',
                'priority': 'info'
            })
        
        return recommendations