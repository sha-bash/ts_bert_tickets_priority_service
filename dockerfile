# Этап 1: Установка зависимостей и загрузка данных NLTK
FROM python:3.11.8-slim-bookworm AS builder

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev \
    unzip \
    libopenblas64-0 \
    libomp5 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Создание виртуального окружения
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Загрузка данных NLTK
RUN python -m nltk.downloader stopwords punkt && \
    mkdir -p /opt/nltk_data && mv /root/nltk_data/* /opt/nltk_data/

# -----------------------------------------------------------
# Этап 2: Финальный образ
FROM python:3.11.8-slim-bookworm

# Настройка переменных окружения
ENV NLTK_DATA=/opt/nltk_data \
    PATH="/opt/venv/bin:$PATH" \
    OMP_NUM_THREADS=1

# Установка runtime-зависимостей
RUN apt-get update && apt-get install -y \
    libopenblas64-0 \
    libomp5 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя и рабочих директорий
RUN useradd -m appuser && \
    mkdir -p /app /app/ml_service && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Копирование виртуального окружения и данных
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /opt/nltk_data /opt/nltk_data

# Копирование исходного кода
COPY --chown=appuser:appuser ./api_gateway/app .
# Копирование моделей
COPY --chown=appuser:appuser ./ml_service/app/models/prioritization_model /app/ml_service/models/prioritization_model
COPY --chown=appuser:appuser ./ml_service/app/models/saved_models /app/ml_service/models/saved_models

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]