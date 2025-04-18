import os
import re
import asyncio
import logging
from functools import partial
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from pymystem3 import Mystem

logger = logging.getLogger(__name__)

class AsyncModelLoader:
    def __init__(self):
        self.prioritization_pipeline = None
        self.retry_count = 3
        self.retry_delay = 5
        self.mystem = Mystem()

    async def _load_model_async(self, model_path, task, num_labels):
        loop = asyncio.get_event_loop()
        try:
            model = await loop.run_in_executor(
                None,
                partial(
                    BertForSequenceClassification.from_pretrained,
                    model_path,
                    num_labels=num_labels,
                    local_files_only=True
                )
            )
            tokenizer = await loop.run_in_executor(
                None,
                partial(BertTokenizer.from_pretrained, model_path, local_files_only=True)
            )
            return pipeline(task, model=model, tokenizer=tokenizer)
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            raise

    async def initialize(self):
        model_path = os.getenv('MODEL_PATH', "/app/ml_service/models/prioritization_model")
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

    async def process_text(self, text: str, threshold: float = 0.70):
        if not self.prioritization_pipeline:
            raise RuntimeError("Model not initialized")

        text = re.sub(r"\s+", " ", text.lower()).strip()
        text = re.sub(r"[^\w\s'\\:]+", "", text)
        lemmatized_text = ''.join(self.mystem.lemmatize(text))

        loop = asyncio.get_event_loop()
        priority_result = await loop.run_in_executor(
            None,
            partial(self.prioritization_pipeline, lemmatized_text)
        )

        logger.info(f"Raw prediction result: {priority_result}")

        if not isinstance(priority_result, list) or not all(isinstance(x, dict) for x in priority_result):
            raise ValueError(f"Unexpected prediction format: {priority_result}")

        score_label_1 = next((x["score"] for x in priority_result if x["label"] in ("1", "LABEL_1")), 0.0)
        predicted_label = "LABEL_1" if score_label_1 >= threshold else "LABEL_0"

        return {
            "task_priority": predicted_label,
            "priority_score": score_label_1
        }

model_loader = AsyncModelLoader()
