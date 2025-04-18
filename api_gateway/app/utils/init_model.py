from utils.model_loader import model_loader
import asyncio
import logging

logger = logging.getLogger(__name__)

def init_model_sync():
    """Синхронная инициализация модели для Celery worker"""
    logger.info("Starting sync model initialization (Celery)...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(model_loader.initialize())
    loop.close()
    logger.info("Model initialized successfully (Celery)")
