from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        try:
            from api.recognition_views import initialize_models
            logger.info("Запуск ініціалізації моделі...")
            
            import threading
            
            def init_models_thread():
                try:
                    success = initialize_models()
                    if success:
                        logger.info("Модель успішно ініціалізована при запуску сервера")
                    else:
                        logger.warning("Помилка ініціалізації моделі при запуску")
                except Exception as e:
                    logger.error(f"Критична помилка ініціалізації моделі: {e}")
            
            timer = threading.Timer(2.0, init_models_thread)
            timer.start()
            
        except Exception as e:
            logger.error(f"Помилка при ініціалізації моделей в apps.py: {e}")