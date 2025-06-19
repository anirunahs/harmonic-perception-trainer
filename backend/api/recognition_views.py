import os
import time
import numpy as np
import logging
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework import serializers

from ..core.audio_processing import RecognitionAudioProcessor
from ..core.feature_extraction import FFTFeatureExtractor
from ..core.model_inference import IntervalClassifier

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
        self.processor = RecognitionAudioProcessor()
        self.feature_extractor = FFTFeatureExtractor()
        self.classifier = IntervalClassifier()
        
    def post(self, request):
        """Аналіз аудіо записи"""
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
        import base64
        
        audio_b64 = validated_data['audio_data']
        audio_data = base64.b64decode(audio_b64)
        
        audio = self.processor.load_audio_from_blob(audio_data)
        
        preprocessing_level = validated_data.get('preprocessing_level', 'standard')
        if preprocessing_level == 'aggressive':
            self.processor.noise_gate_threshold = -50
        elif preprocessing_level == 'minimal':
            self.processor.noise_gate_threshold = -70
        
        processing_result = self.processor.process(audio)
        
        if not processing_result.segments:
            return {
                'error': 'Не знайдено музичні сегменти в записі',
                'warnings': processing_result.warnings,
                'quality_score': processing_result.quality_score
            }
        
        max_segments = validated_data.get('max_segments', 3)
        segments_to_analyze = processing_result.segments[:max_segments]
        
        recognition_results = []
        
        for i, segment in enumerate(segments_to_analyze):
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
        
        final_result = self._finalize_recognition_result(
            recognition_results, 
            processing_result,
            validated_data
        )
        
        return final_result
    
    def _finalize_recognition_result(self, recognition_results, processing_result, validated_data):
        """Формування підсумкового результату"""
        
        if not recognition_results:
            return {
                'status': 'error',
                'message': 'Не вдалося розпізнати інтервали',
                'processing_info': {
                    'duration': processing_result.original_duration,
                    'processing_time': processing_result.processing_time,
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
                'processing_time': processing_result.processing_time,
                'quality_score': processing_result.quality_score,
                'warnings': processing_result.warnings,
                'preprocessing_level': validated_data.get('preprocessing_level', 'standard')
            },
            'recommendations': self._generate_recommendations(recognition_results, processing_result)
        }
    
    def _generate_recommendations(self, recognition_results, processing_result):
        """Генерація рекомендацій для користувача"""
        recommendations = []
        
        avg_audio_quality = np.mean([r['quality_metrics']['audio_quality'] 
                                   for r in recognition_results])
        avg_prediction_confidence = np.mean([r['quality_metrics']['prediction_confidence'] 
                                           for r in recognition_results])
        
        if avg_audio_quality < 0.5:
            recommendations.append({
                'type': 'audio_quality',
                'message': 'Низька якість аудіо. Спробуйте записати в тишому місці.',
                'severity': 'warning'
            })
        
        if avg_prediction_confidence < 0.6:
            recommendations.append({
                'type': 'recognition',
                'message': 'Низька впевненість розпізнавання. Переконайтеся, що граєте чіткі інтервали.',
                'severity': 'info'
            })
        
        if processing_result.quality_score < 0.4:
            recommendations.append({
                'type': 'recording',
                'message': 'Спробуйте записати інтервали повільніше та чіткіше.',
                'severity': 'warning'
            })
        
        if len(recognition_results) == 1 and recognition_results[0]['duration'] < 1.5:
            recommendations.append({
                'type': 'duration',
                'message': 'Зробіть довший запис для кращого аналізу.',
                'severity': 'info'
            })
        
        return recommendations

class QuickRecognitionView(APIView):
    """Швидке розпізнавання з мінімальною обробкою"""
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Швидкий аналіз (тестування)"""
        serializer = AudioRecognitionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            import base64
            start_time = time.time()
            
            audio_b64 = serializer.validated_data['audio_data']
            audio_data = base64.b64decode(audio_b64)
            
            processor = RecognitionAudioProcessor()
            processor.noise_gate_threshold = -70
            
            audio = processor.load_audio_from_blob(audio_data)
            
            audio = processor.remove_dc_offset(audio)
            audio = processor.normalize_rms(audio)
            
            target_samples = int(2.0 * processor.sr)
            if len(audio) >= target_samples:
                start = (len(audio) - target_samples) // 2
                audio_segment = audio[start:start + target_samples]
            else:
                audio_segment = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
            
            feature_extractor = FFTFeatureExtractor()
            features = feature_extractor.extract_features(audio_segment)
            
            classifier = IntervalClassifier()
            prediction = classifier.predict(features)
            
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