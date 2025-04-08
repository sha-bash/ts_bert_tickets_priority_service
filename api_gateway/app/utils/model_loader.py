from transformers import BertForSequenceClassification, BertTokenizer, pipeline

# Универсальная загрузка моделей
def create_pipeline(model_path: str, task: str = "text-classification"):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return pipeline(task, model=model, tokenizer=tokenizer)

# Инициализация предобработчика и моделей
classification_pipeline = create_pipeline("/app/ml_service/models/saved_models")
prioritization_pipeline = create_pipeline("/app/ml_service/models/prioritization_model")

# Пайплайн обработки
def process_text(text: str):
    class_result = classification_pipeline(text)[0]
    priority_result = prioritization_pipeline(text)[0]
    return {
        "task_class": class_result["label"],
        "class_score": class_result["score"],
        "task_priority": priority_result["label"],
        "priority_score": priority_result["score"]
    }


def process_text_prioritization(text: str):
    priority_result = prioritization_pipeline(text)[0]
    return {
        "task_priority": priority_result["label"],
        "priority_score": priority_result["score"]
    }