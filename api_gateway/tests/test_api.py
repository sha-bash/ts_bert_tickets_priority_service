import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)

def test_status(client):
    response = client.get("/status")
    assert response.status_code == 200  
    assert response.json() == {"status": "API is running"} 

def test_bert_prediction(client):
    payload = {"text": "Пример текста для анализа"}
    response = client.post("/v1/bert_prediction", json=payload)
    assert response.status_code == 200
    assert "BERT_predict" in response.json()