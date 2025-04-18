from utils.celery_app import celery
from utils.init_model import init_model_sync
import utils.tasks
import logging

logger = logging.getLogger(__name__)

logger.info("Starting Celery worker initialization...")

try:
    init_model_sync()
    logger.info("Worker initialization complete.")
except Exception as e:
    logger.critical(f"Worker failed to initialize: {e}")
    raise

app = celery
