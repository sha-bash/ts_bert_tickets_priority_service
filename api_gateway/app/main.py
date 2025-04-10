import logging
import asyncio
import crud
import models
import schema 
from models import Session
from fastapi import FastAPI, HTTPException
from fastapi_cache.decorator import cache
from lifispan import lifespan
from dependencies import SessionDependency
from utils.model_loader import model_loader
from async_timeout import timeout


logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title='Сервис для приоритезации обращений второй линии',
    version='1.0.1',
    description='Сервис использует алгоритмы глубокого обучения для приоритезации обращения',
    lifespan=lifespan
)

@app.get("/status")
async def status():
    return {"status": "API is running"}
    
@app.post('/v1/bert_prediction', response_model=schema.CreateBERTResponse)
@cache(expire=300)
async def add_predict(
    predict_json: schema.CreatePredictRequest,
    session: SessionDependency 
):
    '''
    Асинхронная функция принимает на вход строку обращения и производит ее приоритезацию с помощью модели BERT.

    Responses:

        200: Успешный ответ (см. CreateBERTPriorizationResponse).

        400: Неверный запрос (пустой текст или слишком длинный).
        
        500: Ошибка сервера (модель или БД недоступны).
    '''

    try:
        text = str(predict_json.text)
        logging.info(f"Received text: {text}")

        result = await model_loader.process_text(text)
        logging.info(f"BERT prediction: {result}")

        # Создание объекта для БД
        db_prediction = models.Prediction(
            text_task=text,
            task_priority=result["task_priority"],
            priority_score=result["priority_score"]
        )

        # Сохранение в БД
        async with timeout(10):
            await crud.add_prediction(session, db_prediction)

        return result

    except asyncio.TimeoutError:
        logging.error("Таймаут операции с БД")
        raise HTTPException(status_code=504, detail="Таймаут сервиса")

    except Exception as e:
        logging.error(f"Critical error: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")