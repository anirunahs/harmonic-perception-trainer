"""Unit tests for AudioProcessingFacade and RecognitionResult."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import base64


class TestRecognitionResult(unittest.TestCase):
    """Tests for RecognitionResult dataclass."""
    
    def test_error_result(self):
        """Verify error result creation."""
        from core.recognition.result import RecognitionResult
        
        result = RecognitionResult.error_result('Test error', 'test_code')
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, 'Test error')
        self.assertEqual(result.error_code, 'test_code')
    
    def test_from_prediction_success(self):
        """Verify result creation from prediction dict."""
        from core.recognition.result import RecognitionResult
        
        prediction = {
            'best_prediction': {
                'interval': 'perfect_5th',
                'confidence': 0.95
            },
            'predictions': [
                {'interval': 'perfect_5th', 'confidence': 0.95},
                {'interval': 'perfect_4th', 'confidence': 0.03},
                {'interval': 'major_3rd', 'confidence': 0.02}
            ]
        }
        
        result = RecognitionResult.from_prediction(
            prediction,
            processing_time=0.5,
            audio_duration=2.0
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.best_prediction.interval, 'perfect_5th')
        self.assertEqual(result.best_prediction.confidence, 0.95)
        self.assertEqual(len(result.top_predictions), 3)
        self.assertEqual(result.processing_time, 0.5)
    
    def test_from_prediction_error(self):
        """Verify result creation from error prediction."""
        from core.recognition.result import RecognitionResult
        
        prediction = {
            'error': True,
            'message': 'Recognition failed'
        }
        
        result = RecognitionResult.from_prediction(prediction)
        
        self.assertFalse(result.success)
        self.assertIn('failed', result.error)
    
    def test_to_dict_success(self):
        """Verify to_dict for success result."""
        from core.recognition.result import RecognitionResult, PredictionItem
        
        result = RecognitionResult(
            success=True,
            best_prediction=PredictionItem('perfect_5th', 0.95, 1),
            top_predictions=[PredictionItem('perfect_5th', 0.95, 1)],
            processing_time=0.5,
            audio_duration=2.0
        )
        
        d = result.to_dict()
        
        self.assertEqual(d['status'], 'success')
        self.assertIn('best_prediction', d)
        self.assertIn('processing_time', d)
    
    def test_to_dict_error(self):
        """Verify to_dict for error result."""
        from core.recognition.result import RecognitionResult
        
        result = RecognitionResult.error_result('Test error', 'test_code')
        d = result.to_dict()
        
        self.assertEqual(d['status'], 'error')
        self.assertEqual(d['error'], 'Test error')


class TestPredictionItem(unittest.TestCase):
    """Tests for PredictionItem dataclass."""
    
    def test_percentage_property(self):
        """Verify percentage calculation."""
        from core.recognition.result import PredictionItem
        
        item = PredictionItem('test', 0.856, 1)
        
        self.assertEqual(item.percentage, 85.6)
    
    def test_to_dict(self):
        """Verify to_dict method."""
        from core.recognition.result import PredictionItem
        
        item = PredictionItem('perfect_5th', 0.95, 1)
        d = item.to_dict()
        
        self.assertEqual(d['interval'], 'perfect_5th')
        self.assertEqual(d['confidence'], 0.95)
        self.assertEqual(d['percentage'], 95.0)
        self.assertEqual(d['rank'], 1)


class TestAudioProcessingFacade(unittest.TestCase):
    """Tests for AudioProcessingFacade."""
    
    def setUp(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    def tearDown(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_is_ready_property(self, mock_extractor, mock_classifier):
        """Verify is_ready reflects model state."""
        mock_classifier.return_value.is_loaded = True
        
        from core.recognition.facade import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        
        self.assertTrue(facade.is_ready)
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    @patch('core.ml.model_manager.logger')
    def test_recognize_when_model_not_ready(self, mock_logger, mock_extractor, mock_classifier):
        """Verify error when model not loaded."""
        mock_classifier.return_value.is_loaded = False
        
        from core.recognition.facade import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        result = facade.recognize_interval('dGVzdA==')
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, 'model_unavailable')
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    @patch('core.recognition.facade.librosa')
    def test_recognize_interval_success(self, mock_librosa, mock_extractor, mock_classifier):
        """Verify successful recognition."""
        # Setup mocks
        mock_classifier_instance = MagicMock()
        mock_classifier_instance.is_loaded = True
        mock_classifier_instance.predict.return_value = {
            'best_prediction': {'interval': 'perfect_5th', 'confidence': 0.95},
            'predictions': [{'interval': 'perfect_5th', 'confidence': 0.95}]
        }
        mock_classifier.return_value = mock_classifier_instance
        
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_features.return_value = np.zeros(100)
        mock_extractor.return_value = mock_extractor_instance
        
        mock_librosa.load.return_value = (np.random.randn(44100), 22050)
        
        from core.recognition.facade import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        
        # Create valid base64 audio (dummy)
        audio_b64 = base64.b64encode(b'dummy_audio_data').decode()
        result = facade.recognize_interval(audio_b64)
        
        self.assertTrue(result.success)
        self.assertEqual(result.best_prediction.interval, 'perfect_5th')
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    @patch('core.recognition.facade.logger')
    def test_recognize_invalid_base64(self, mock_logger, mock_extractor, mock_classifier):
        """Verify error on invalid base64."""
        mock_classifier.return_value.is_loaded = True
        
        from core.recognition.facade import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        result = facade.recognize_interval('not_valid_base64!!!')
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, 'decode_error')
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    @patch('core.recognition.facade.librosa')
    def test_recognize_empty_audio(self, mock_librosa, mock_extractor, mock_classifier):
        """Verify error on empty audio."""
        mock_classifier.return_value.is_loaded = True
        mock_librosa.load.return_value = (np.array([]), 22050)
        
        from core.recognition.facade import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        audio_b64 = base64.b64encode(b'empty').decode()
        result = facade.recognize_interval(audio_b64)
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, 'empty_audio')


class TestAudioPreprocessing(unittest.TestCase):
    """Tests for audio preprocessing methods."""
    
    def setUp(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    def tearDown(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_preprocess_normalizes_audio(self, mock_extractor, mock_classifier):
        """Verify audio is normalized."""
        mock_classifier.return_value.is_loaded = True
        
        from core.recognition.facade import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        
        # Create audio with DC offset
        audio = np.random.randn(44100) + 5.0
        processed = facade._preprocess_audio(audio, 22050)
        
        # Check DC offset removed (mean close to 0)
        self.assertAlmostEqual(np.mean(processed), 0, delta=0.1)
        
        # Check amplitude normalized
        self.assertLessEqual(np.max(np.abs(processed)), 1.0)
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_extract_best_segment_pads_short_audio(self, mock_extractor, mock_classifier):
        """Verify short audio is padded."""
        mock_classifier.return_value.is_loaded = True
        
        from core.recognition.facade import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        
        # Create short audio (1 second at 22050 Hz)
        short_audio = np.random.randn(22050)
        result = facade._extract_best_segment(short_audio, 22050)
        
        # Should be padded to 2 seconds
        expected_length = int(2.0 * 22050)
        self.assertEqual(len(result), expected_length)
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_extract_best_segment_finds_energy(self, mock_extractor, mock_classifier):
        """Verify best segment is extracted based on energy."""
        mock_classifier.return_value.is_loaded = True
        
        from core.recognition.facade import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        
        # Create audio with high energy in the middle
        sr = 22050
        audio = np.zeros(sr * 5)  # 5 seconds
        audio[sr*2:sr*4] = np.random.randn(sr*2) * 2  # High energy 2-4s
        
        result = facade._extract_best_segment(audio, sr)
        
        # Result should have higher energy than if taken from start
        result_energy = np.mean(result**2)
        start_energy = np.mean(audio[:sr*2]**2)
        
        self.assertGreater(result_energy, start_energy)


class TestAudioQualityAnalysis(unittest.TestCase):
    """Tests for audio quality analysis."""
    
    def setUp(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    def tearDown(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    @patch('core.recognition.facade.librosa')
    def test_analyze_good_quality(self, mock_librosa, mock_extractor, mock_classifier):
        """Verify good quality detection."""
        mock_classifier.return_value.is_loaded = True
        
        # Loud audio (good quality)
        mock_librosa.load.return_value = (np.random.randn(44100) * 0.5, 22050)
        
        from core.recognition.facade import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        audio_b64 = base64.b64encode(b'good_audio').decode()
        result = facade.analyze_audio_quality(audio_b64)
        
        self.assertEqual(result['quality'], 'good')
        self.assertGreater(result['quality_score'], 0.8)
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    @patch('core.recognition.facade.librosa')
    def test_analyze_poor_quality(self, mock_librosa, mock_extractor, mock_classifier):
        """Verify poor quality detection."""
        mock_classifier.return_value.is_loaded = True
        
        # Very quiet audio (poor quality)
        mock_librosa.load.return_value = (np.random.randn(44100) * 0.001, 22050)
        
        from core.recognition.facade import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        audio_b64 = base64.b64encode(b'quiet_audio').decode()
        result = facade.analyze_audio_quality(audio_b64)
        
        self.assertEqual(result['quality'], 'poor')
        self.assertLess(result['quality_score'], 0.5)
        
        # Should have volume recommendation
        rec_types = [r['type'] for r in result['recommendations']]
        self.assertIn('volume', rec_types)


if __name__ == '__main__':
    unittest.main()
