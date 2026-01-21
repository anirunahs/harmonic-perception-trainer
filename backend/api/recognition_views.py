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

from core.ml import ModelManager

logger = logging.getLogger(__name__)

# Ініціалізація Singleton ModelManager при завантаженні модуля
# Модель завантажується негайно (eager loading)
_model_manager = ModelManager.get_instance()

class AudioRecognitionSerializer(serializers.Serializer):
    """Серіалізатор для запиту розпізнавання"""
    audio_data = serializers.CharField(help_text="Base64 encoded audio data")
    format = serializers.ChoiceField(
        choices=['wav', 'webm', 'mp3', 'ogg'],
        default='webm',
        help_text="Audio format"
    )

class IntervalRecognitionView(APIView):
    """Розпізнавання інтервалів"""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def post(self, request):
        """Швидкий аналіз аудіо"""
        # Використовуємо Singleton ModelManager
        model_manager = ModelManager.get_instance()
        
        if not model_manager.is_ready:
            return Response(
                {
                    'error': 'Модель розпізнавання недоступна',
                    'details': model_manager.load_error or 'Система не змогла ініціалізувати модель',
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
                audio = self._quick_processing(audio, sr)
            except Exception as e:
                logger.error(f"Помилка обробки: {e}")
                return Response(
                    {'error': 'Помилка обробки аудіо'}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            try:
                features = model_manager.feature_extractor.extract_features(audio)
                prediction = model_manager.classifier.predict(features)
                
                if prediction.get('error', False):
                    raise Exception(prediction.get('message', 'Помилка розпізнавання'))
                
            except Exception as e:
                logger.error(f"Помилка розпізнавання: {e}")
                return Response(
                    {'error': 'Помилка розпізнавання інтервалу'}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            processing_time = time.time() - start_time
            
            best_prediction = prediction['best_prediction']
            top_predictions = prediction['predictions'][:3]
            
            return Response({
                'status': 'success',
                'best_prediction': {
                    'interval': best_prediction['interval'],
                    'confidence': best_prediction['confidence'],
                    'percentage': round(best_prediction['confidence'] * 100, 1)
                },
                'top_predictions': [
                    {
                        'interval': pred['interval'],
                        'confidence': pred['confidence'],
                        'percentage': round(pred['confidence'] * 100, 1),
                        'rank': i + 1
                    }
                    for i, pred in enumerate(top_predictions)
                ],
                'processing_time': round(processing_time, 2),
                'audio_duration': round(len(audio) / sr, 1)
            })
            
        except Exception as e:
            logger.error(f"Критична помилка розпізнавання: {e}")
            return Response(
                {'error': 'Системна помилка розпізнавання'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _quick_processing(self, audio, sr):
        """Швидка обробка аудіо"""
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
            # Використовуємо Singleton ModelManager
            model_manager = ModelManager.get_instance()
            manager_status = model_manager.get_status()
            
            model_info = manager_status.get('model_info', {'loaded': False})
            
            return Response({
                'status': 'ready' if model_manager.is_ready else 'error',
                'models_loaded': model_manager.is_ready,
                'load_error': model_manager.load_error,
                'components': {
                    'model': {
                        'status': 'ready' if manager_status['components']['classifier']['loaded'] else 'error',
                        'info': model_info
                    },
                    'feature_extractor': {
                        'status': 'ready' if manager_status['components']['feature_extractor']['loaded'] else 'error'
                    }
                },
                'system_info': {
                    'tensorflow_version': model_info.get('tensorflow_version', 'unknown'),
                    'supported_intervals': [
                        'major_2nd', 'major_3rd', 'major_6th', 'major_7th', 
                        'minor_2nd', 'minor_3rd', 'minor_6th', 'minor_7th', 
                        'perfect_4th', 'perfect_5th', 'perfect_8th', 'tritone'
                    ]
                },
                'available_endpoints': [
                    '/api/recognition/interval/',
                    '/api/recognition/status/',
                    '/api/recognition/diagnostics/'
                ]
            })
            
        except Exception as e:
            logger.error(f"Помилка перевірки статусу: {e}")
            return Response({
                'status': 'error',
                'error': str(e),
                'models_loaded': False,
                'components': {
                    'model': {'status': 'error'},
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
                quality_score = 0.9
            elif rms_db > -40:
                quality = 'fair'
                quality_score = 0.6
            else:
                quality = 'poor'
                quality_score = 0.3
            
            recommendations = []
            if rms_db < -40:
                recommendations.append({
                    'type': 'volume',
                    'message': f'Тихий запис ({rms_db:.1f}dB). Записуйте ближче до мікрофона.',
                    'priority': 'high'
                })
            
            if duration < 1.5:
                recommendations.append({
                    'type': 'duration',
                    'message': 'Короткий запис. Рекомендована тривалість: 2-3 секунди.',
                    'priority': 'medium'
                })
            
            if peak > 0.99:
                recommendations.append({
                    'type': 'clipping',
                    'message': 'Виявлено кліпування. Зменшіть гучність запису.',
                    'priority': 'high'
                })
            
            if not recommendations:
                recommendations.append({
                    'type': 'quality',
                    'message': 'Хороша якість запису для розпізнавання.',
                    'priority': 'info'
                })
            
            return Response({
                'status': 'success',
                'diagnostics': {
                    'duration': round(duration, 2),
                    'rms_db': round(rms_db, 1),
                    'peak_db': round(peak_db, 1),
                    'quality': quality,
                    'quality_score': quality_score,
                    'recommendations': recommendations
                }
            })
            
        except Exception as e:
            logger.error(f"Помилка діагностики: {e}")
            return Response(
                {'error': 'Помилка аналізу аудіо'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )