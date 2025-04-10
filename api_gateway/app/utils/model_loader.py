from transformers import BertForSequenceClassification, BertTokenizer, pipeline
import re
import asyncio
from functools import partial
import logging

logger = logging.getLogger(__name__)

class AsyncModelLoader:
    def __init__(self):
        self.prioritization_pipeline = None
        self.retry_count = 3
        self.retry_delay = 5

    async def _load_model_async(self, model_path, task, num_labels):
        loop = asyncio.get_event_loop()
        try:
            # Загрузка модели в отдельном потоке
            model = await loop.run_in_executor(
                None,
                partial(
                    BertForSequenceClassification.from_pretrained,
                    model_path,
                    num_labels=num_labels
                )
            )
            
            # Загрузка токенизатора
            tokenizer = await loop.run_in_executor(
                None,
                partial(BertTokenizer.from_pretrained, model_path)
            )
            
            return pipeline(task, model=model, tokenizer=tokenizer)
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            raise

    async def initialize(self):
        """Асинхронная инициализация с повторными попытками"""
        model_path = "/app/ml_service/models/prioritization_model"
        for attempt in range(self.retry_count):
            try:
                self.prioritization_pipeline = await self._load_model_async(
                    model_path=model_path,
                    task="text-classification",
                    num_labels=2
                )
                logger.info("Model loaded successfully")
                return
            except Exception as e:
                if attempt == self.retry_count - 1:
                    raise RuntimeError(f"Failed to load model after {self.retry_count} attempts")
                logger.warning(f"Retrying model load in {self.retry_delay} seconds... (Attempt {attempt+1}/{self.retry_count})")
                await asyncio.sleep(self.retry_delay)

    async def process_text(self, text: str):
        """Асинхронная обработка текста"""
        if not self.prioritization_pipeline:
            raise RuntimeError("Model not initialized")
            
        # Предобработка текста
        text = re.sub(r"\s+", " ", text).strip()
        cleaned_text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", "", text)
        
        # Запуск предсказания в отдельном потоке
        loop = asyncio.get_event_loop()
        priority_result = await loop.run_in_executor(
            None,
            partial(self.prioritization_pipeline, cleaned_text))
        
        return {
            "task_priority": priority_result[0]["label"],
            "priority_score": priority_result[0]["score"]
        }

# Инициализация асинхронного загрузчика
model_loader = AsyncModelLoader()