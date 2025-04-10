import os
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

async def setup_cache():
    """Асинхронная инициализация Redis кеша"""
    redis_url = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
    
    # Асинхронное подключение
    redis = await aioredis.from_url(
        redis_url,
        encoding="utf8",
        decode_responses=True
    )
    
    # Инициализация кеша
    FastAPICache.init(RedisBackend(redis), prefix="bert-cache")