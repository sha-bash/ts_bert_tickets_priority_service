from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from peft import PeftModel, PeftConfig
import os


def load_lora_model(base_model_name: str, lora_path: str, merge_lora: bool = True, num_labels: int = 2):
    """Загрузка LoRA-модели с опциональным объединением весов."""

    base_model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
    model = PeftModel.from_pretrained(base_model, lora_path)
    if merge_lora:
        model = model.merge_and_unload()
    return model

def create_pipeline(
    model_path: str,
    task: str = "text-classification",
    is_lora: bool = False,
    base_model_name: str = None,
    merge_lora: bool = True,
    num_labels: int = None
):
    """Универсальная загрузка пайплайна с поддержкой LoRA."""
    try:
        if is_lora:
            if not base_model_name:
                raise ValueError("Для LoRA укажите base_model_name (например, 'DeepPavlov/rubert-base-cased')")
            
            # Проверка наличия файлов адаптера
            required_files = ['adapter_config.json']
            if not all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
                raise FileNotFoundError(f"Не найдены файлы адаптера LoRA в {model_path}")
            
            model = load_lora_model(base_model_name, model_path, merge_lora, num_labels=num_labels)
            tokenizer = BertTokenizer.from_pretrained(base_model_name)
        else:
            model = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels
            )
            tokenizer = BertTokenizer.from_pretrained(model_path)
        
        return pipeline(task, model=model, tokenizer=tokenizer)
    
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели: {str(e)}")

# Инициализация моделей 
try:
    prioritization_pipeline = create_pipeline(
        "/app/ml_service/models/custom_prioritization_model",
        is_lora=False,
        num_labels=2
    )
    
    custom_prioritization_pipeline = create_pipeline(
        "/app/ml_service/models/prioritization_model",
        is_lora=True,
        base_model_name="DeepPavlov/rubert-base-cased",
        merge_lora=True,
        num_labels=2
    )
except Exception as e:
    print(f"CRITICAL ERROR: Не удалось загрузить модели. {str(e)}")
    raise


# Пайплайн обработки
def process_text(text: str):
    class_result = custom_prioritization_pipeline(text)[0]
    priority_result = prioritization_pipeline(text)[0]
    return {
        "custom_task_priority": class_result["label"],
        "custom_priority_score": class_result["score"],
        "task_priority": priority_result["label"],
        "priority_score": priority_result["score"]
    }
