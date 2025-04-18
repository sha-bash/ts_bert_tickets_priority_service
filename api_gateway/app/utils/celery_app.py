from celery import Celery
import os

celery = Celery(
    "worker",
    broker=f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/0",
    backend=f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/1",
    include=["utils.tasks"]
)

celery.conf.update(
    task_track_started=True,
    result_expires=3600,  # 1 час хранения результатов
)