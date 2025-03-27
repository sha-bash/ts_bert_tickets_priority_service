from pydantic import BaseModel


class CreatePredictRequest(BaseModel):
    text: str

class CreateBERTResponse(BaseModel):
    task_class: str
    class_score: float  
    task_priority: str  
    priority_score: float
