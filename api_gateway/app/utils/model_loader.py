from preprocess import TextPreprocessor
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

# Универсальная загрузка моделей
def create_pipeline(model_path: str, task: str = "text-classification"):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return pipeline(task, model=model, tokenizer=tokenizer)

# Инициализация предобработчика и моделей
text_preprocessor = TextPreprocessor()
classification_pipeline = create_pipeline("models/classification_model")
prioritization_pipeline = create_pipeline("models/prioritization_model")

# Пайплайн обработки
def process_text(text: str):
    cleaned_text = text_preprocessor.preprocess_text(text)
    class_result = classification_pipeline(cleaned_text)[0]
    priority_result = prioritization_pipeline(cleaned_text)[0]
    return {
        "task_class": class_result["label"],
        "class_score": class_result["score"],
        "task_priority": priority_result["label"],
        "priority_score": priority_result["score"]
    }