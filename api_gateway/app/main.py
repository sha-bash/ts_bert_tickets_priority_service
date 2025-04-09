import logging
import asyncio
import crud
import models
from models import Session
from fastapi import FastAPI, HTTPException
from lifispan import lifespan
from dependencies import SessionDependency
from utils.model_loader import process_text
import shema as shema


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
    
@app.post('/v1/bert_prediction', response_model=shema.CreateBERTResponse)
async def add_predict(
    predict_json: shema.CreatePredictRequest,
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

        result = process_text(text)
        logging.info(f"BERT prediction: {result}")

        # Создание объекта для БД
        db_prediction = models.Prediction(
            custom_task_priority=result["custom_task_priority"],
            custom_task_score=result["custom_priority_score"],
            task_priority=result["task_priority"],
            priority_score=result["priority_score"]
        )

        # Сохранение в БД
        await crud.add_prediction(session, db_prediction)

        return result
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    