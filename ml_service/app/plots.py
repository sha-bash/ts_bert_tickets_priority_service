import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


logger = logging.getLogger(__name__)

def plot_class_distribution(numeric_labels, PLOTS_DIR):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=numeric_labels)
    plt.title('Распределение классов')
    plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'))
    plt.close()

def plot_metrics(metrics_history, PLOTS_DIR):
    """Визуализирует метрики обучения"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(metrics_history['loss'], label='Training Loss')
    plt.plot(metrics_history['eval_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(metrics_history['accuracy'], label='Training Accuracy')
    plt.plot(metrics_history['eval_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(metrics_history['f1'], label='Training F1')
    plt.plot(metrics_history['eval_f1'], label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Метрики обучения сохранены в {plot_path}")

def plot_confusion_matrix(y_true, y_pred, classes, PLOTS_DIR):
    """Визуализирует матрицу ошибок"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Матрица ошибок сохранена в {plot_path}")