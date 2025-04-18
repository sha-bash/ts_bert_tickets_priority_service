from celery import Celery
import os
import logging

logger = logging.getLogger(__name__)

def make_celery():
    redis_host = os.getenv("REDIS_HOST")
    redis_port = os.getenv("REDIS_PORT")

    if not redis_host or not redis_port:
        raise ValueError("Missing Redis config")

    return Celery(
        "worker",
        broker=f"redis://{redis_host}:{redis_port}/0",
        backend=f"redis://{redis_host}:{redis_port}/1",
        include=["utils.tasks"]
    )

celery = make_celery()

celery.conf.update(
    task_track_started=True,
    result_expires=3600,
)
