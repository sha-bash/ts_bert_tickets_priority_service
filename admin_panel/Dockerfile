# Используем легковесный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY requirements.txt .
COPY app.py ./

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт Streamlit
EXPOSE 8501

# Запуск Streamlit-приложения
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
