"""Unit tests for feature extraction strategies."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np


class TestFeatureExtractionStrategy(unittest.TestCase):
    """Tests for base FeatureExtractionStrategy."""
    
    def test_is_abstract(self):
        """Verify base class cannot be instantiated."""
        from core.recognition.strategies.base import FeatureExtractionStrategy
        
        with self.assertRaises(TypeError):
            FeatureExtractionStrategy()
    
    def test_abstract_methods(self):
        """Verify abstract methods are defined."""
        from core.recognition.strategies.base import FeatureExtractionStrategy
        import abc
        
        # Check abstract methods exist
        abstract_methods = FeatureExtractionStrategy.__abstractmethods__
        self.assertIn('extract_features', abstract_methods)
        self.assertIn('get_name', abstract_methods)
        self.assertIn('get_feature_count', abstract_methods)


class TestFFTStrategy(unittest.TestCase):
    """Tests for FFTStrategy."""
    
    def test_initialization_defaults(self):
        """Verify default initialization."""
        from core.recognition.strategies import FFTStrategy
        
        strategy = FFTStrategy()
        
        self.assertEqual(strategy._fft_size, 2048)
        self.assertEqual(strategy._sample_rate, 22050)
    
    def test_initialization_custom(self):
        """Verify custom initialization."""
        from core.recognition.strategies import FFTStrategy
        
        strategy = FFTStrategy(fft_size=4096, sample_rate=44100)
        
        self.assertEqual(strategy._fft_size, 4096)
        self.assertEqual(strategy._sample_rate, 44100)
    
    def test_get_name(self):
        """Verify strategy name."""
        from core.recognition.strategies import FFTStrategy
        
        strategy = FFTStrategy()
        
        self.assertEqual(strategy.get_name(), "FFT-based")
    
    def test_get_feature_count(self):
        """Verify feature count is positive."""
        from core.recognition.strategies import FFTStrategy
        
        strategy = FFTStrategy()
        count = strategy.get_feature_count()
        
        self.assertGreater(count, 0)
        self.assertIsInstance(count, int)
    
    def test_extract_features_returns_array(self):
        """Verify extract_features returns numpy array."""
        from core.recognition.strategies import FFTStrategy
        
        strategy = FFTStrategy()
        audio = np.random.randn(22050)  # 1 second
        
        features = strategy.extract_features(audio, 22050)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
    
    def test_extract_features_short_audio(self):
        """Verify handling of short audio."""
        from core.recognition.strategies import FFTStrategy
        
        strategy = FFTStrategy()
        short_audio = np.random.randn(1000)  # Very short
        
        features = strategy.extract_features(short_audio, 22050)
        
        self.assertIsInstance(features, np.ndarray)
    
    def test_extract_features_long_audio(self):
        """Verify handling of long audio."""
        from core.recognition.strategies import FFTStrategy
        
        strategy = FFTStrategy()
        long_audio = np.random.randn(88200)  # 4 seconds
        
        features = strategy.extract_features(long_audio, 22050)
        
        self.assertIsInstance(features, np.ndarray)
    
    def test_extract_features_sine_wave(self):
        """Verify features from sine wave."""
        from core.recognition.strategies import FFTStrategy
        
        strategy = FFTStrategy()
        
        # Generate 440Hz sine wave
        sr = 22050
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * 440 * t)
        
        features = strategy.extract_features(audio, sr)
        
        # Should extract meaningful features
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(np.max(np.abs(features)), 0)
    
    def test_get_info(self):
        """Verify get_info method."""
        from core.recognition.strategies import FFTStrategy
        
        strategy = FFTStrategy()
        info = strategy.get_info()
        
        self.assertIn('name', info)
        self.assertIn('feature_count', info)
        self.assertIn('type', info)
        self.assertEqual(info['name'], 'FFT-based')
        self.assertEqual(info['type'], 'FFTStrategy')


class TestMFCCStrategy(unittest.TestCase):
    """Tests for MFCCStrategy."""
    
    def test_initialization_defaults(self):
        """Verify default initialization."""
        from core.recognition.strategies import MFCCStrategy
        
        strategy = MFCCStrategy()
        
        self.assertEqual(strategy._n_mfcc, 13)
        self.assertEqual(strategy._n_mels, 128)
    
    def test_initialization_custom(self):
        """Verify custom initialization."""
        from core.recognition.strategies import MFCCStrategy
        
        strategy = MFCCStrategy(n_mfcc=20, n_mels=64)
        
        self.assertEqual(strategy._n_mfcc, 20)
        self.assertEqual(strategy._n_mels, 64)
    
    def test_get_name(self):
        """Verify strategy name."""
        from core.recognition.strategies import MFCCStrategy
        
        strategy = MFCCStrategy()
        
        self.assertEqual(strategy.get_name(), "MFCC-based")
    
    def test_get_feature_count(self):
        """Verify feature count."""
        from core.recognition.strategies import MFCCStrategy
        
        strategy = MFCCStrategy()
        count = strategy.get_feature_count()
        
        # 13 MFCCs * 6 (mean, std, delta, delta2) + 7 contrast + 12 chroma = 97
        self.assertEqual(count, 97)
    
    def test_extract_features_returns_array(self):
        """Verify extract_features returns numpy array."""
        from core.recognition.strategies import MFCCStrategy
        
        strategy = MFCCStrategy()
        audio = np.random.randn(22050)  # 1 second
        
        features = strategy.extract_features(audio, 22050)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), strategy.get_feature_count())
    
    def test_extract_features_sine_wave(self):
        """Verify features from sine wave."""
        from core.recognition.strategies import MFCCStrategy
        
        strategy = MFCCStrategy()
        
        # Generate 440Hz sine wave
        sr = 22050
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * 440 * t)
        
        features = strategy.extract_features(audio, sr)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), strategy.get_feature_count())


class TestStrategyComparison(unittest.TestCase):
    """Tests for comparing strategies."""
    
    def test_different_feature_counts(self):
        """Verify strategies produce different feature counts."""
        from core.recognition.strategies import FFTStrategy, MFCCStrategy
        
        fft = FFTStrategy()
        mfcc = MFCCStrategy()
        
        # They should have different feature counts
        self.assertNotEqual(fft.get_feature_count(), mfcc.get_feature_count())
    
    def test_different_names(self):
        """Verify strategies have different names."""
        from core.recognition.strategies import FFTStrategy, MFCCStrategy
        
        fft = FFTStrategy()
        mfcc = MFCCStrategy()
        
        self.assertNotEqual(fft.get_name(), mfcc.get_name())
    
    def test_both_extract_from_same_audio(self):
        """Verify both strategies can process same audio."""
        from core.recognition.strategies import FFTStrategy, MFCCStrategy
        
        fft = FFTStrategy()
        mfcc = MFCCStrategy()
        
        audio = np.random.randn(22050)
        sr = 22050
        
        fft_features = fft.extract_features(audio, sr)
        mfcc_features = mfcc.extract_features(audio, sr)
        
        # Both should return arrays
        self.assertIsInstance(fft_features, np.ndarray)
        self.assertIsInstance(mfcc_features, np.ndarray)
        
        # With different lengths
        self.assertNotEqual(len(fft_features), len(mfcc_features))


class TestStrategyWithFacade(unittest.TestCase):
    """Tests for Strategy integration with Facade."""
    
    def setUp(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    def tearDown(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    @patch('core.ml.model_manager.IntervalClassifier')
    @patch('core.ml.model_manager.FFTFeatureExtractor')
    def test_facade_default_strategy(self, mock_extractor, mock_classifier):
        """Verify Facade uses FFTStrategy by default."""
        mock_classifier.return_value.is_loaded = True
        
        from core.recognition import AudioProcessingFacade, FFTStrategy
        
        facade = AudioProcessingFacade()
        
        self.assertIsInstance(facade.strategy, FFTStrategy)
        self.assertEqual(facade.strategy_name, "FFT-based")
    
    @patch('core.ml.model_manager.IntervalClassifier')
    @patch('core.ml.model_manager.FFTFeatureExtractor')
    def test_facade_custom_strategy(self, mock_extractor, mock_classifier):
        """Verify Facade accepts custom strategy."""
        mock_classifier.return_value.is_loaded = True
        
        from core.recognition import AudioProcessingFacade, MFCCStrategy
        
        facade = AudioProcessingFacade(strategy=MFCCStrategy())
        
        self.assertIsInstance(facade.strategy, MFCCStrategy)
        self.assertEqual(facade.strategy_name, "MFCC-based")
    
    @patch('core.ml.model_manager.IntervalClassifier')
    @patch('core.ml.model_manager.FFTFeatureExtractor')
    def test_facade_set_strategy(self, mock_extractor, mock_classifier):
        """Verify strategy can be changed dynamically."""
        mock_classifier.return_value.is_loaded = True
        
        from core.recognition import AudioProcessingFacade, FFTStrategy, MFCCStrategy
        
        facade = AudioProcessingFacade()
        self.assertEqual(facade.strategy_name, "FFT-based")
        
        facade.set_strategy(MFCCStrategy())
        self.assertEqual(facade.strategy_name, "MFCC-based")
        
        facade.set_strategy(FFTStrategy())
        self.assertEqual(facade.strategy_name, "FFT-based")
    
    @patch('core.ml.model_manager.IntervalClassifier')
    @patch('core.ml.model_manager.FFTFeatureExtractor')
    def test_facade_set_invalid_strategy(self, mock_extractor, mock_classifier):
        """Verify error on invalid strategy."""
        mock_classifier.return_value.is_loaded = True
        
        from core.recognition import AudioProcessingFacade
        
        facade = AudioProcessingFacade()
        
        with self.assertRaises(TypeError):
            facade.set_strategy("not a strategy")
        
        with self.assertRaises(TypeError):
            facade.set_strategy(None)
    
    @patch('core.ml.model_manager.IntervalClassifier')
    @patch('core.ml.model_manager.FFTFeatureExtractor')
    @patch('core.recognition.facade.librosa')
    def test_facade_compare_strategies(self, mock_librosa, mock_extractor, mock_classifier):
        """Verify compare_strategies method."""
        mock_classifier.return_value.is_loaded = True
        mock_librosa.load.return_value = (np.random.randn(22050), 22050)
        
        # Mock librosa.feature functions for MFCC
        mock_librosa.feature.mfcc.return_value = np.random.randn(13, 100)
        mock_librosa.feature.delta.return_value = np.random.randn(13, 100)
        mock_librosa.feature.spectral_contrast.return_value = np.random.randn(7, 100)
        mock_librosa.feature.chroma_stft.return_value = np.random.randn(12, 100)
        
        from core.recognition import AudioProcessingFacade, FFTStrategy, MFCCStrategy
        import base64
        
        facade = AudioProcessingFacade()
        audio_b64 = base64.b64encode(b'test_audio').decode()
        
        results = facade.compare_strategies(
            audio_b64,
            [FFTStrategy(), MFCCStrategy()]
        )
        
        self.assertIn('FFT-based', results)
        self.assertIn('MFCC-based', results)
        self.assertTrue(results['FFT-based']['success'])


if __name__ == '__main__':
    unittest.main()
