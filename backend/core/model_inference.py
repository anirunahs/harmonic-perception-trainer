import numpy as np
import joblib
import os
from typing import Dict, Optional
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class IntervalClassifier:

    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(settings.BASE_DIR, 'ml_models', 'fft_interval_model.keras')
        self.scaler_path = scaler_path or os.path.join(settings.BASE_DIR, 'ml_models', 'fft_scaler.joblib')
        
        self.model = None
        self.scaler = None
        self.is_loaded = False
        
        self.class_names = [
            'major_2nd', 'major_3rd', 'major_6th', 'major_7th',
            'minor_2nd', 'minor_3rd', 'minor_6th', 'minor_7th',
            'perfect_4th', 'perfect_5th', 'perfect_8th', 'tritone'
        ]
        
        logger.info(f"Ініціалізація:")
        for i, name in enumerate(self.class_names):
            logger.info(f"  {i}: {name}")
        
        self.load_model()
    
    def load_model(self) -> bool:
        """Завантаження моделі"""
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Scaler завантажено успішно")
            else:
                logger.error(f"Scaler не знайдено: {self.scaler_path}")
                return False
            
            if os.path.exists(self.model_path):
                try:
                    import tensorflow as tf
                    
                    tf.get_logger().setLevel('ERROR')
                    
                    self.model = tf.keras.models.load_model(self.model_path, compile=False)
                    
                    self.model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    logger.info("Модель завантажена")
                    logger.info(f"Вхідна форма: {self.model.input_shape}")
                    logger.info(f"Вихідна форма: {self.model.output_shape}")
                    
                    self.is_loaded = True
                    return True
                    
                except Exception as e:
                    logger.error(f"Помилка завантаження моделі: {e}")
                    return False
            else:
                logger.error(f"Файл моделі не знайдено: {self.model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Критична помилка завантаження моделі: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Dict:
        """Прогнозування"""
        if not self.is_loaded:
            return {
                'error': True,
                'message': 'Модель не завантажена',
                'best_prediction': {'interval': 'unknown', 'confidence': 0.0, 'quality': 'error'},
                'predictions': []
            }
        
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            logger.info(f"Розмір вхідних ознак: {features.shape}")
            
            try:
                features_scaled = self.scaler.transform(features)
                logger.info("Ознаки масштабовані успішно")
            except Exception as e:
                logger.warning(f"Помилка масштабування: {e}")
                features_scaled = features
            
            logger.info("Запуск прогнозування...")
            predictions = self.model.predict(features_scaled, verbose=0)
            prediction_probs = predictions[0] if predictions.ndim > 1 else predictions
            
            logger.info(f"Отримано прогнози: {prediction_probs.shape}")
            logger.info(f"Топ 3 ймовірності: {np.sort(prediction_probs)[-3:]}")
            
            best_index = np.argmax(prediction_probs)
            best_confidence = float(prediction_probs[best_index])
            best_interval = self.class_names[best_index]
            
            top_indices = np.argsort(prediction_probs)[-5:][::-1]
            predictions_list = []
            
            for i, idx in enumerate(top_indices):
                interval_name = self.class_names[idx]
                confidence = float(prediction_probs[idx])
                
                predictions_list.append({
                    'interval': interval_name,
                    'confidence': confidence,
                    'rank': i + 1
                })
                
                logger.info(f"  #{i+1}: {interval_name} - {confidence:.3f}")
            
            quality_level = 'high' if best_confidence > 0.7 else 'medium' if best_confidence > 0.5 else 'low'
            
            result = {
                'best_prediction': {
                    'interval': best_interval,
                    'confidence': best_confidence,
                    'quality': quality_level
                },
                'predictions': predictions_list,
                'quality_analysis': {
                    'confidence_level': quality_level,
                    'max_confidence': float(np.max(prediction_probs)),
                    'is_reliable': best_confidence > 0.5,
                    'entropy': self._calculate_entropy(prediction_probs)
                },
                'model_info': {
                    'model_type': 'FFT Neural Network',
                    'classes_count': len(self.class_names),
                    'input_shape': features_scaled.shape[1]
                }
            }
            
            logger.info(f"Найкращий прогноз: {best_interval} ({best_confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Помилка прогнозування: {e}")
            return {
                'error': True,
                'message': str(e),
                'best_prediction': {'interval': 'unknown', 'confidence': 0.0, 'quality': 'error'},
                'predictions': []
            }
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Розрахунок ентропії для оцінки невизначеності"""
        try:
            probs = probabilities / np.sum(probabilities)
            probs = np.maximum(probs, 1e-10)
            entropy = -np.sum(probs * np.log(probs))
            return float(entropy)
        except Exception:
            return 0.0
    
    def get_model_info(self) -> Dict:
        """Інформація про модель"""
        info = {
            'loaded': self.is_loaded,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'classes': self.class_names,
            'classes_count': len(self.class_names),
            'files_exist': {
                'model': os.path.exists(self.model_path),
                'scaler': os.path.exists(self.scaler_path)
            }
        }
        
        if self.is_loaded and self.model:
            try:
                import tensorflow as tf
                info.update({
                    'tensorflow_version': tf.__version__,
                    'model_params': self.model.count_params(),
                    'input_shape': str(self.model.input_shape),
                    'output_shape': str(self.model.output_shape)
                })
            except Exception as e:
                logger.warning(f"Не вдалося отримати додаткову інформацію про модель: {e}")
        
        return info