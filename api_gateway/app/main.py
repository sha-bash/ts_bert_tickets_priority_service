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