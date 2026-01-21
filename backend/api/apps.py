from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        """Initialize ML models on server startup using ModelManager Singleton."""
        try:
            import threading
            
            def init_models_thread():
                try:
                    from core.ml import ModelManager
                    
                    logger.info("Initializing ML models via ModelManager...")
                    manager = ModelManager.get_instance()
                    
                    if manager.is_ready:
                        logger.info("ML models initialized successfully on server startup")
                    else:
                        logger.warning(f"ML model initialization failed: {manager.load_error}")
                except Exception as e:
                    logger.error(f"Critical error initializing ML models: {e}")
            
            # Delayed initialization to avoid Django startup issues
            timer = threading.Timer(2.0, init_models_thread)
            timer.start()
            
        except Exception as e:
            logger.error(f"Error in apps.py ready(): {e}")
