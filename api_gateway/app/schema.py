from pydantic import BaseModel, Field


class CreatePredictRequest(BaseModel):
    text: str = Field(..., max_length=1000, example="Текст обращения")


class CreateBERTResponse(BaseModel):
    task_priority: str  
    priority_score: float
