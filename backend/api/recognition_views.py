"""
Recognition Views - API endpoints for interval recognition.

Uses AudioProcessingFacade for simplified audio processing.
"""

import logging
from rest_framework import status, serializers
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

from core.recognition import AudioProcessingFacade

logger = logging.getLogger(__name__)


class AudioRecognitionSerializer(serializers.Serializer):
    """Serializer for recognition request."""
    audio_data = serializers.CharField(help_text="Base64 encoded audio data")
    format = serializers.ChoiceField(
        choices=['wav', 'webm', 'mp3', 'ogg'],
        default='webm',
        help_text="Audio format"
    )


class IntervalRecognitionView(APIView):
    """Interval recognition endpoint."""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def post(self, request):
        """Recognize interval from audio."""
        serializer = AudioRecognitionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Use Facade for recognition
        facade = AudioProcessingFacade()
        
        if not facade.is_ready:
            return Response(
                {
                    'error': 'Recognition model unavailable',
                    'details': facade.model_error or 'Model not initialized',
                    'status': 'service_unavailable'
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        # Single method call handles everything
        result = facade.recognize_interval(serializer.validated_data['audio_data'])
        
        if not result.success:
            error_status = status.HTTP_500_INTERNAL_SERVER_ERROR
            if result.error_code in ('decode_error', 'empty_audio'):
                error_status = status.HTTP_400_BAD_REQUEST
            
            return Response(
                {'error': result.error, 'error_code': result.error_code},
                status=error_status
            )
        
        return Response(result.to_dict())


class ModelStatusView(APIView):
    """Model status endpoint."""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def get(self, request):
        """Get system status."""
        try:
            facade = AudioProcessingFacade()
            
            # Get status from ModelManager through Facade
            from core.ml import ModelManager
            model_manager = ModelManager.get_instance()
            manager_status = model_manager.get_status()
            model_info = manager_status.get('model_info', {'loaded': False})
            
            return Response({
                'status': 'ready' if facade.is_ready else 'error',
                'models_loaded': facade.is_ready,
                'load_error': facade.model_error,
                'components': {
                    'model': {
                        'status': 'ready' if manager_status['components']['classifier']['loaded'] else 'error',
                        'info': model_info
                    },
                    'feature_extractor': {
                        'status': 'ready' if manager_status['components']['feature_extractor']['loaded'] else 'error'
                    },
                    'facade': {
                        'status': 'ready',
                        'type': 'AudioProcessingFacade'
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
            logger.error(f"Status check error: {e}")
            return Response({
                'status': 'error',
                'error': str(e),
                'models_loaded': False
            })


class AudioDiagnosticsView(APIView):
    """Audio diagnostics endpoint."""
    
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    
    def post(self, request):
        """Analyze audio quality."""
        audio_data = request.data.get('audio_data', '')
        
        if not audio_data:
            return Response(
                {'error': 'No audio data provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Use Facade for diagnostics
        facade = AudioProcessingFacade()
        diagnostics = facade.analyze_audio_quality(audio_data)
        
        if 'error' in diagnostics:
            return Response(
                {'error': diagnostics['error']},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response({
            'status': 'success',
            'diagnostics': diagnostics
        })
