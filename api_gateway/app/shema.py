from pydantic import BaseModel


class CreatePredictRequest(BaseModel):
    text: str

class CreateBERTResponse(BaseModel):
    custom_task_priority: str  
    custom_priority_score: float 
    task_priority: str  
    priority_score: float
