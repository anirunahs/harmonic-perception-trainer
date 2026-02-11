"""Unit tests for ModelManager Singleton."""

import unittest
from unittest.mock import patch, MagicMock
import threading


class TestModelManagerSingleton(unittest.TestCase):
    """Tests for Singleton behavior."""
    
    def setUp(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    def tearDown(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_singleton_returns_same_instance(self, mock_extractor, mock_classifier):
        """Verify same instance is returned."""
        mock_classifier.return_value.is_loaded = True
        
        from core.ml.model_manager import ModelManager
        
        instance1 = ModelManager.get_instance()
        instance2 = ModelManager.get_instance()
        instance3 = ModelManager()
        
        self.assertIs(instance1, instance2)
        self.assertIs(instance2, instance3)
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_thread_safety(self, mock_extractor, mock_classifier):
        """Verify thread-safe creation."""
        mock_classifier.return_value.is_loaded = True
        
        from core.ml.model_manager import ModelManager
        
        instances = []
        errors = []
        
        def create_instance():
            try:
                instance = ModelManager.get_instance()
                instances.append(instance)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(instances), 10)
        
        first_instance = instances[0]
        for instance in instances[1:]:
            self.assertIs(instance, first_instance)


class TestModelManagerLoading(unittest.TestCase):
    """Tests for model loading."""
    
    def setUp(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    def tearDown(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_eager_loading_on_creation(self, mock_extractor, mock_classifier):
        """Verify models are loaded immediately on creation."""
        mock_classifier.return_value.is_loaded = True
        
        from core.ml.model_manager import ModelManager
        
        instance = ModelManager.get_instance()
        
        mock_extractor.assert_called_once()
        mock_classifier.assert_called_once()
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_is_ready_when_loaded(self, mock_extractor, mock_classifier):
        """Verify is_ready returns True when loaded."""
        mock_classifier.return_value.is_loaded = True
        
        from core.ml.model_manager import ModelManager
        
        instance = ModelManager.get_instance()
        
        self.assertTrue(instance.is_ready)
        self.assertIsNone(instance.load_error)
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_is_not_ready_when_classifier_not_loaded(self, mock_extractor, mock_classifier):
        """Verify is_ready returns False when classifier not loaded."""
        mock_classifier.return_value.is_loaded = False
        
        from core.ml.model_manager import ModelManager
        
        instance = ModelManager.get_instance()
        
        self.assertFalse(instance.is_ready)
        self.assertIsNotNone(instance.load_error)
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_handles_loading_exception(self, mock_extractor, mock_classifier):
        """Verify exception handling during load."""
        mock_classifier.side_effect = Exception("Test error")
        
        from core.ml.model_manager import ModelManager
        
        instance = ModelManager.get_instance()
        
        self.assertFalse(instance.is_ready)
        self.assertIn("Test error", instance.load_error)


class TestModelManagerAccess(unittest.TestCase):
    """Tests for component access."""
    
    def setUp(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    def tearDown(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_classifier_property(self, mock_extractor, mock_classifier):
        """Verify classifier property returns classifier."""
        mock_classifier_instance = MagicMock()
        mock_classifier_instance.is_loaded = True
        mock_classifier.return_value = mock_classifier_instance
        
        from core.ml.model_manager import ModelManager
        
        instance = ModelManager.get_instance()
        
        self.assertIs(instance.classifier, mock_classifier_instance)
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_feature_extractor_property(self, mock_extractor, mock_classifier):
        """Verify feature_extractor property returns extractor."""
        mock_extractor_instance = MagicMock()
        mock_extractor.return_value = mock_extractor_instance
        mock_classifier.return_value.is_loaded = True
        
        from core.ml.model_manager import ModelManager
        
        instance = ModelManager.get_instance()
        
        self.assertIs(instance.feature_extractor, mock_extractor_instance)
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_classifier_raises_when_not_loaded(self, mock_extractor, mock_classifier):
        """Verify RuntimeError when accessing unloaded classifier."""
        mock_classifier.return_value.is_loaded = False
        
        from core.ml.model_manager import ModelManager
        
        instance = ModelManager.get_instance()
        
        with self.assertRaises(RuntimeError):
            _ = instance.classifier
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_get_status(self, mock_extractor, mock_classifier):
        """Verify get_status returns dict with correct keys."""
        mock_classifier.return_value.is_loaded = True
        mock_classifier.return_value.get_model_info.return_value = {'version': '1.0'}
        
        from core.ml.model_manager import ModelManager
        
        instance = ModelManager.get_instance()
        status = instance.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('is_ready', status)
        self.assertIn('components', status)
        self.assertTrue(status['is_ready'])


class TestModelManagerReload(unittest.TestCase):
    """Tests for model reloading."""
    
    def setUp(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    def tearDown(self):
        from core.ml.model_manager import ModelManager
        ModelManager.reset_instance()
    
    @patch('core.model_inference.IntervalClassifier')
    @patch('core.feature_extraction.FFTFeatureExtractor')
    def test_reload_reloads_models(self, mock_extractor, mock_classifier):
        """Verify reload() reloads models."""
        mock_classifier.return_value.is_loaded = True
        
        from core.ml.model_manager import ModelManager
        
        instance = ModelManager.get_instance()
        
        self.assertEqual(mock_classifier.call_count, 1)
        
        result = instance.reload()
        
        self.assertTrue(result)
        self.assertEqual(mock_classifier.call_count, 2)
        self.assertEqual(mock_extractor.call_count, 2)


if __name__ == '__main__':
    unittest.main()
