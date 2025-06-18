import numpy as np
import joblib
import tensorflow as tf
import json
import os
from typing import Dict, Optional
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class IntervalClassifier:
    """Клас для інференсу моделі розпізнавання інтервалів"""
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        self.model_path = model_path or self._get_default_model_path()
        self.scaler_path = scaler_path or self._get_default_scaler_path()
        
        self.model = None
        self.scaler = None
        self.class_names = []
        self.is_loaded = False
        
        self.load_model()
    
    def _get_default_model_path(self) -> str:
        """Отримання шляху до моделі за замовчуванням"""
        return os.path.join(settings.BASE_DIR, 'ml_models', 'fft_interval_model.keras')
    
    def _get_default_scaler_path(self) -> str:
        """Отримання шляху до scaler за замовчуванням"""
        return os.path.join(settings.BASE_DIR, 'ml_models', 'fft_scaler.joblib')
    
    def load_model(self) -> bool:
        """Завантаження моделі та метаданих"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Модель не знайдена: {self.model_path}")
                return False
            
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Модель завантажена: {self.model_path}")
            
            if not os.path.exists(self.scaler_path):
                logger.error(f"Scaler не знайдено: {self.scaler_path}")
                return False
            
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler завантажено: {self.scaler_path}")
            
            self._load_metadata()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Помилка завантаження моделі: {e}")
            return False
    
    def _load_metadata(self):
        """Завантаження метаданих моделі"""
        metadata_path = self.model_path.replace('.keras', '_metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.class_names = metadata.get('class_names', [])
                logger.info(f"Завантажено {len(self.class_names)} класів з метаданих")
                
            except Exception as e:
                logger.warning(f"Помилка завантаження метаданих: {e}")
                self._set_default_class_names()
        else:
            logger.warning("Метадані не знайдені, використовуються базові назви класів")
            self._set_default_class_names()
    
    def _set_default_class_names(self):
        """Встановлення базових назв класів"""
        self.class_names = [
            'minor_2nd', 'major_2nd', 'minor_3rd', 'major_3rd', 
            'perfect_4th', 'tritone', 'perfect_5th', 'minor_6th', 
            'major_6th', 'minor_7th', 'major_7th', 'perfect_8th'
        ]
    
    def predict(self, features: np.ndarray) -> Dict:
        """Прогнозування інтервалу з ознак"""
        if not self.is_loaded:
            raise RuntimeError("Модель не завантажена")
        
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            features_scaled = self.scaler.transform(features)
            
            predictions = self.model.predict(features_scaled, verbose=0)
            prediction_probs = predictions[0]
            
            return self._format_prediction_result(prediction_probs)
            
        except Exception as e:
            logger.error(f"Помилка прогнозування: {e}")
            return self._get_error_result(str(e))
    
    def _format_prediction_result(self, prediction_probs: np.ndarray) -> Dict:
        """Форматування результату прогнозування"""
        best_index = np.argmax(prediction_probs)
        best_confidence = float(prediction_probs[best_index])
        best_interval = self.class_names[best_index]
        
        top_indices = np.argsort(prediction_probs)[-5:][::-1]
        predictions = []
        
        for idx in top_indices:
            predictions.append({
                'interval': self.class_names[idx],
                'confidence': float(prediction_probs[idx]),
                'rank': len(predictions) + 1
            })
        
        quality_analysis = self._analyze_prediction_quality(prediction_probs)
        
        return {
            'best_prediction': {
                'interval': best_interval,
                'confidence': best_confidence,
                'quality': quality_analysis['confidence_level']
            },
            'predictions': predictions,
            'quality_analysis': quality_analysis,
            'model_info': {
                'model_type': 'FFT Neural Network',
                'classes_count': len(self.class_names),
                'confidence_threshold': 0.7
            }
        }
    
    def _analyze_prediction_quality(self, prediction_probs: np.ndarray) -> Dict:
        """Аналіз якості прогнозування"""
        max_confidence = np.max(prediction_probs)
        second_max = np.partition(prediction_probs, -2)[-2]
        
        confidence_gap = max_confidence - second_max
        
        entropy = -np.sum(prediction_probs * np.log(prediction_probs + 1e-10))
        max_entropy = np.log(len(prediction_probs))
        normalized_entropy = entropy / max_entropy
        
        if max_confidence >= 0.8 and confidence_gap >= 0.3:
            confidence_level = 'high'
        elif max_confidence >= 0.6 and confidence_gap >= 0.2:
            confidence_level = 'medium'
        elif max_confidence >= 0.4:
            confidence_level = 'low'
        else:
            confidence_level = 'very_low'
        
        return {
            'confidence_level': confidence_level,
            'max_confidence': float(max_confidence),
            'confidence_gap': float(confidence_gap),
            'entropy': float(normalized_entropy),
            'certainty_score': float(1.0 - normalized_entropy),
            'is_reliable': max_confidence >= 0.6 and confidence_gap >= 0.2
        }
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Результат з помилкою"""
        return {
            'error': True,
            'message': error_message,
            'best_prediction': {
                'interval': 'unknown',
                'confidence': 0.0,
                'quality': 'error'
            },
            'predictions': [],
            'quality_analysis': {
                'confidence_level': 'error',
                'is_reliable': False
            }
        }
    
    def get_model_info(self) -> Dict:
        """Інформація про модель"""
        if not self.is_loaded:
            return {'loaded': False, 'error': 'Модель не завантажена'}
        
        return {
            'loaded': True,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'classes': self.class_names,
            'classes_count': len(self.class_names),
            'model_params': self.model.count_params() if self.model else 0,
            'input_shape': self.model.input_shape if self.model else None
        }