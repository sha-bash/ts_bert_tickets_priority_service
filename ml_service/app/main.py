import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import json
import logging
import yaml
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from plots import plot_metrics, plot_confusion_matrix, plot_class_distribution
from transformers import (
    BertModel,
    BertPreTrainedModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from peft import LoraConfig, get_peft_model # Добавил для уменьшения вычислительных затрат при fine-tuning

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Константы
#DATA_PATH = os.path.join('data', 'processed', 'merged_requests_without_dubl.csv')
DATA_PATH = os.path.join('ml_service','data', 'processed', 'processed_requests_nodubl.csv')
MODEL_CONFIG_PATH = os.path.join('ml_service', 'app','configs', 'multiclass_config.yaml')
SAVED_MODEL_AND_TOKENIZER_PATH = os.path.join('ml_service', 'app','models', 'saved_models')
LABEL_MAPPING_PATH = os.path.join(SAVED_MODEL_AND_TOKENIZER_PATH, 'label_mapping.json')
METRICS_PATH = os.path.join(SAVED_MODEL_AND_TOKENIZER_PATH, 'metrics.json')
PLOTS_DIR = os.path.join('ml_service', 'app','models', 'plots')
LOGGING_DIR = os.path.join('ml_service', 'app','models', 'logging')
RESULTS_DIR = os.path.join('ml_service', 'app','models', 'results')

# Создание директорий
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODEL_AND_TOKENIZER_PATH, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class CustomBERTClassifier(BertPreTrainedModel):
    """Кастомная модель классификатора на основе BERT"""
    def __init__(self, config, num_classes, dropout_rate=0.2):
        super().__init__(config)
        self.bert = BertModel(config)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        
        if input_ids is not None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            
        pooled_output = outputs.last_hidden_state[:, 0]
        norm_out = self.norm(pooled_output)
        dropped_out = self.dropout(norm_out)
        logits = self.classifier(dropped_out)
        
        loss = None
        if labels is not None:
            loss_fct = FocalLoss() 
            loss = loss_fct(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits
        }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss_fct = FocalLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

class MetricLoggerCallback(TrainerCallback):
    """Кастомный колбэк для логирования метрик"""
    def __init__(self, metrics_history):
        self.metrics_history = metrics_history
        self.trainer = None  
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key in self.metrics_history.keys():
                if key in logs:
                    self.metrics_history[key].append(logs[key])

    def on_train_begin(self, args, state, control, **kwargs):
        """Автоматически вызывается при старте обучения"""
        logger.info("Начало обучения")
        logger.info(f"Используемое устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        if 'trainer' in kwargs:
            self.trainer = kwargs['trainer']
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        """Изменение learning rate после 5-й эпохи"""
        if state.epoch == 5 and self.trainer:
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = 5e-6
            logger.info("Learning rate уменьшен до 5e-6")

class FocalLoss(nn.Module):
    """Focal Loss для работы с несбалансированными классами"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def read_yaml(file_path):
    """Загрузка конфигурации с валидацией параметров"""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Валидация структуры конфига
    required_sections = ['model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Отсутствует обязательная секция: {section}")

    # Параметры модели
    model_config = config['model']
    if 'name' not in model_config:
        raise ValueError("Для модели должно быть указано имя (name)")

    # Параметры обучения
    training_config = config['training']
    
    # Автоматическое преобразование типов
    type_conversion = {
        'int': [
            'num_train_epochs', 'per_device_train_batch_size',
            'per_device_eval_batch_size', 'gradient_accumulation_steps',
            'warmup_steps', 'logging_steps', 'eval_steps', 'max_seq_length'
        ],
        'float': [
            'learning_rate', 'weight_decay', 'warmup_ratio',
            'label_smoothing_factor'
        ],
        'bool': ['fp16', 'gradient_checkpointing', 'group_by_length']
    }

    for param in training_config:
        value = training_config[param]
        if param in type_conversion['int']:
            training_config[param] = int(value)
        elif param in type_conversion['float']:
            training_config[param] = float(value)
        elif param in type_conversion['bool']:
            training_config[param] = bool(value)
    
    peft_config = config['peft']
    if 'target_modules' not in peft_config:
        raise ValueError("Для LORA должны быть указаны слои BERT для адаптации)")
    
    # Значения по умолчанию для опциональных параметров
    defaults = {
        'max_seq_length': 256,
        'gradient_accumulation_steps': 1,
        'label_smoothing_factor': 0.0
    }
    
    for param, default in defaults.items():
        if param not in training_config:
            training_config[param] = default
            logger.warning(f"Установлено значение по умолчанию для {param}: {default}")

    return config

class TextDataset(torch.utils.data.Dataset):
    """Кастомный Dataset для обработки текстовых данных"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def main():

    # Загрузка и подготовка данных
    logger.info("Загрузка и подготовка данных...")
    try:
        df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')
        logger.info(f"Загружено {len(df)} записей")
        
        # Исходные метки (могут быть числами или строками)
        original_labels = df['Метка'].values
        
        # Преобразование меток в числовой формат [0, N-1]
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(original_labels)
        
        # Сохранение маппинга
        label_mapping = {int(i): str(label) for i, label in enumerate(label_encoder.classes_)}
        with open(LABEL_MAPPING_PATH, 'w') as f:
            json.dump(label_mapping, f, indent=4)
        
        logger.info(f"Маппинг меток сохранен в {LABEL_MAPPING_PATH}")
        logger.info(f"Соответствие меток: {label_mapping}")
        
        # Анализ распределения классов
        label_counts = pd.Series(numeric_labels).value_counts().sort_index()
        label_counts.index = [f"{label_mapping[i]} ({i})" for i in label_counts.index]
        logger.info("Распределение классов:\n" + str(label_counts))
        
        # Проверка на наличие классов с малым количеством примеров
        MIN_SAMPLES_PER_CLASS = 2
        rare_classes = [i for i, count in enumerate(np.bincount(numeric_labels)) if count < MIN_SAMPLES_PER_CLASS]
        
        if rare_classes:
            rare_classes_names = [label_mapping[i] for i in rare_classes]
            logger.warning(f"Найдены классы с малым количеством примеров: {dict(zip(rare_classes, rare_classes_names))}")
            mask = ~np.isin(numeric_labels, rare_classes)
            df = df[mask]
            numeric_labels = numeric_labels[mask]
            logger.info(f"Удалены примеры редких классов. Осталось {len(df)} записей")
            
            # Обновляем маппинг после удаления редких классов
            remaining_labels = np.unique(numeric_labels)
            label_mapping = {new: label_mapping[old] for new, old in enumerate(sorted(remaining_labels))}
            with open(LABEL_MAPPING_PATH, 'w') as f:
                json.dump(label_mapping, f, indent=4)
            
            logger.info(f"Обновленный маппинг меток: {label_mapping}")
        
        texts = df['Текст'].fillna('').astype(str)
        
        if len(df) == 0:
            raise ValueError("После фильтрации не осталось данных для обучения")
        
        # Разделение данных
        if len(label_mapping) > 1:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels
            )
        else:
            logger.warning("Остался только один класс - стратификация невозможна")
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, numeric_labels, test_size=0.2, random_state=42
            )
        
        logger.info(f"Разделение данных: train={len(train_texts)}, val={len(val_texts)}")
        
        # Логирование распределения классов
        train_label_counts = pd.Series(train_labels).value_counts().sort_index()
        train_label_counts.index = [f"{label_mapping[i]} ({i})" for i in train_label_counts.index]
        logger.info("Распределение классов в обучающей выборке:\n" + str(train_label_counts))
        
        val_label_counts = pd.Series(val_labels).value_counts().sort_index()
        val_label_counts.index = [f"{label_mapping[i]} ({i})" for i in val_label_counts.index]
        logger.info("Распределение классов в валидационной выборке:\n" + str(val_label_counts))

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        raise

    # Загрузка конфигурации
    logger.info("Загрузка конфигурации модели...")
    try:
        config = read_yaml(MODEL_CONFIG_PATH)
        model_config = config['model']
        training_config = config['training']
        peft_config = config['peft']

        max_steps = training_config.get('max_steps', -1)
        optim = training_config.get('optim', 'adamw_torch')
        
        training_args = TrainingArguments(
            output_dir=RESULTS_DIR,
            # Основные параметры
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            
            # Оптимизация обучения
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            fp16=training_config['fp16'],
            gradient_checkpointing=training_config['gradient_checkpointing'],
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
            warmup_ratio=training_config['warmup_ratio'],
            group_by_length=training_config.get('group_by_length', False),
            label_smoothing_factor=training_config['label_smoothing_factor'],
            max_steps=max_steps,
            optim=optim,
            dataloader_num_workers=2,
            save_total_limit=1,
            save_steps=1000,
            no_cuda=True,
            
            # Мониторинг
            logging_dir=LOGGING_DIR,
            logging_steps=training_config['logging_steps'],
            eval_strategy=training_config['eval_strategy'],
            eval_steps=training_config.get('eval_steps', 500),
            save_strategy=training_config['save_strategy'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            greater_is_better=training_config['greater_is_better'],
            report_to="none"
        )

    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {str(e)}", exc_info=True)
        raise

    # Токенизация
    logger.info("Инициализация токенизатора...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'], 
        do_lower_case=False,  
        use_fast=True,
        strip_accents=False 
    )

        def tokenize_data(texts):
            return tokenizer(
                texts.tolist(),
                padding='longest',
                truncation=True,
                max_length=min(training_config['max_seq_length'], 512),
                return_tensors="pt"
            )
        
        train_encodings = tokenize_data(train_texts)
        val_encodings = tokenize_data(val_texts)

    except Exception as e:
        logger.error(f"Ошибка при токенизации: {str(e)}", exc_info=True)
        raise

    try:
        train_dataset = TextDataset(train_encodings, train_labels)
        val_dataset = TextDataset(val_encodings, val_labels)
    except Exception as e:
        logger.error(f"Ошибка при создании Dataset: {str(e)}", exc_info=True)
        raise

    def compute_metrics(p):
        """Вычисляет метрики качества модели"""
        preds = p.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            p.label_ids, preds, average='weighted'
        )
        acc = accuracy_score(p.label_ids, preds)
        
        report = classification_report(
            p.label_ids, preds, 
            target_names=[label_mapping[i] for i in sorted(label_mapping.keys())],
            output_dict=True
        )
        
        with open(METRICS_PATH, 'w') as f:
            json.dump(report, f, indent=4)
        
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    
    lora_config = LoraConfig(
        r=peft_config['r'],              
        lora_alpha=peft_config['alpha'],    
        target_modules=peft_config['target_modules'],  
        lora_dropout=peft_config['dropout'],
        bias=peft_config['bias'],
        task_type=peft_config['task_type']  
    )

    # Инициализация модели
    logger.info("Инициализация модели BERT...")
    try:
        model = CustomBERTClassifier.from_pretrained(
            model_config['name'],
            num_classes=len(label_mapping),
            dropout_rate=model_config.get('dropout_rate', 0.3)
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.config.id2label = label_mapping
        model.config.label2id = {v: k for k, v in label_mapping.items()}

    except Exception as e:
        logger.error(f"Ошибка при инициализации модели: {str(e)}", exc_info=True)
        raise

    # Обучение модели
    logger.info("Обучение модели...")
    try:
        metrics_history = {
            'loss': [], 'accuracy': [], 'f1': [],
            'eval_loss': [], 'eval_accuracy': [], 'eval_f1': []
        }
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[MetricLoggerCallback(metrics_history)]
    )
        
        
        trainer.train()
        logger.info("Обучение завершено успешно!")
        
        # Сохранение результатов
        plot_metrics(metrics_history, PLOTS_DIR)
        predictions = trainer.predict(val_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        plot_confusion_matrix(val_labels, preds, [label_mapping[i] for i in sorted(label_mapping.keys())], PLOTS_DIR)
        
        model.save_pretrained(SAVED_MODEL_AND_TOKENIZER_PATH) 
        tokenizer.save_pretrained(SAVED_MODEL_AND_TOKENIZER_PATH)
        logger.info(f"Модель сохранена в {SAVED_MODEL_AND_TOKENIZER_PATH}")

    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}", exc_info=True)
        raise

    logger.info("Все этапы завершены успешно!")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()