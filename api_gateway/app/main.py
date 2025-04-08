import logging
import asyncio
import crud
import models
from models import Session
from fastapi import FastAPI, HTTPException
from lifispan import lifespan
from dependencies import SessionDependency
from utils.model_loader import process_text, process_text_prioritization
import shema as shema


logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title='Сервис для классификации и приоритезации обращений второй линии',
    version='1.0.1',
    description='Сервис использует алгоритмы глубокого обучения для классификации и приоритезации обращения',
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
    Ассинхронная функция принимает на вход строку обращения и производит ее классификацию и приоритезацию с помощью модели BERT
    '''

    try:
        text = str(predict_json.text)
        logging.info(f"Received text: {text}")

        result = process_text(text)
        logging.info(f"BERT prediction: {result}")

        # Создание объекта для БД
        db_prediction = models.Prediction(
            task_class=result["task_class"],
            class_score=result["class_score"],
            task_priority=result["task_priority"],
            priority_score=result["priority_score"]
        )

        # Сохранение в БД
        await crud.add_prediction(session, db_prediction)

        return result
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

app.post('/v1/bert_priorization', response_model=shema.CreateBERTPriorizationResponse)
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

        result = process_text_prioritization(text)
        if not all(key in result for key in ["task_priority", "priority_score"]):
            raise HTTPException(status_code=500, detail="Model returned invalid format")
        
        logging.info(f"BERT prediction: {result}")

        # Создание объекта для БД
        db_prediction = models.Prediction(
            task_class= None,
            class_score= None,
            task_priority=result["task_priority"],
            priority_score=result["priority_score"]
        )

        # Сохранение в БД
        await crud.add_prediction(session, db_prediction)

        return result
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))