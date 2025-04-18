from utils.celery_app import celery
from utils.model_loader import model_loader
import crud
import models
import schema
import logging
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from utils.db import async_session_factory 
from async_timeout import timeout
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    logger.info("Initializing model in Celery worker...")
    asyncio.run(model_loader.initialize())
except Exception as e:
    logger.critical(f"Failed to initialize model in Celery worker: {str(e)}")
    raise

@celery.task(bind=True, name="process_text_task", autoretry_for=(Exception,), retry_backoff=True)
def process_text_task(self, predict_data: dict):
    """Celery-таск: классификация + сохранение в БД с замером времени."""
    text = predict_data["text"]

    try:
        start_time = datetime.now(timezone.utc)
        self.update_state(state='STARTED', meta={'start_time': str(start_time)})

        # Запускаем асинхронную логику синхронно
        result = asyncio.run(_process_and_save(text))

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        return {
            "result": result,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "duration_seconds": duration
        }

    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise self.retry(exc=e, countdown=60)

async def _process_and_save(text: str):
    logger.info(f"Received text: {text}")
    result = await model_loader.process_text(text)

    async with async_session_factory() as session: 
        async with timeout(10):
            db_prediction = models.Prediction(
                text_task=text,
                task_priority=result["task_priority"],
                priority_score=result["priority_score"]
            )
            await crud.add_prediction(session, db_prediction)

    return result
