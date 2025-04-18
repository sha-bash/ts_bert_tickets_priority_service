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
    wget \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Создание виртуального окружения
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копируем requirements.txt раньше кода, чтобы использовать кэш
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Загрузка данных NLTK
RUN python -m nltk.downloader stopwords punkt && \
    mkdir -p /opt/nltk_data && mv /root/nltk_data/* /opt/nltk_data/

# -------------------------------------------------------------------------
# Этап 2: Финальный образ
FROM python:3.11.8-slim-bookworm

# Настройка переменных окружения
ENV NLTK_DATA=/opt/nltk_data \
    PATH="/opt/venv/bin:$PATH" \
    OMP_NUM_THREADS=1 \
    MODEL_PATH=/app/ml_service/models/prioritization_model \
    MYSTEM_BIN=/usr/local/bin/mystem

# Установка runtime-зависимостей + mystem
RUN apt-get update && apt-get install -y \
    libopenblas64-0 \
    libomp5 \
    wget \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://download.cdn.yandex.net/mystem/mystem-3.1-linux-64bit.tar.gz && \
    tar -xvzf mystem-3.1-linux-64bit.tar.gz && \
    mv mystem /usr/local/bin/ && chmod +x /usr/local/bin/mystem && \
    rm mystem-3.1-linux-64bit.tar.gz

# Создание пользователя и директорий
RUN useradd -m appuser && \
    mkdir -p /app /app/ml_service && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Копирование окружения и NLTK
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /opt/nltk_data /opt/nltk_data

# Копирование кода и модели
COPY --chown=appuser:appuser ./api_gateway/app .
COPY --chown=appuser:appuser ./ml_service/app/models/prioritization_model /app/ml_service/models/prioritization_model

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
