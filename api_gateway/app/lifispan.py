from contextlib import asynccontextmanager
from fastapi import FastAPI
from models import Base, engine
from utils.model_loader import model_loader
from utils.cache import setup_cache
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Унифицированный обработчик жизненного цикла"""
    try:
        # Инициализация БД
        logger.info("Initializing database...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Инициализация кеша
        logger.info("Initializing Redis cache...")
        await setup_cache()  
        
        # Загрузка моделей
        logger.info("Loading ML models...")
        await model_loader.initialize()
        
        yield
        
    except Exception as e:
        logger.critical(f"Application bootstrap failed: {str(e)}")
        raise
    
    finally:
        logger.info("Shutting down...")
        await engine.dispose()