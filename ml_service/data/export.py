import os
import pandas as pd
import logging
from tqdm import tqdm
from db.db import DatabaseManager
from ml_service.data.preprocess import TextPreprocessor


os.makedirs("data/logs", exist_ok=True)
logging.basicConfig(
    filename="data/logs/preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_DATA_PATH = "data/raw/requests.csv"
PROCESSED_DATA_PATH = "data/processed/processed_requests.csv"

def load_data_from_db():
    """
    Загружает данные из базы данных и сохраняет их в data/raw/requests.csv.
    
    Returns:
        pd.DataFrame: Данные в формате DataFrame.
    """
    db_manager = DatabaseManager()
    logging.info("Файл не найден, выгружаем данные из БД...")

    try:
        data = db_manager.get_requests_data()
        if not data:
            logging.warning("Нет данных для экспорта.")
            return None

        df = pd.DataFrame(data)
        try:
            df.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8", sep=';')
            logging.info(f"Данные сохранены в {RAW_DATA_PATH}")
        except IOError as e:
            logging.error(f"Ошибка записи в файл {RAW_DATA_PATH}: {e}")
            raise

        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных из БД: {e}")
        return None
    finally:
        db_manager.close()

def preprocess_and_save_data(df: pd.DataFrame):
    preprocessor = TextPreprocessor()
    logging.info("Начинаем предобработку данных...")

    # Проверка наличия столбцов
    if "Заголовок" not in df.columns or "Сообщение" not in df.columns:
        logging.error("Отсутствуют необходимые столбцы 'Заголовок' или 'Сообщение'.")
        return

    # Создаем директорию для обработанных данных, если её нет
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    # Открываем файл для записи
    with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as output_file:
        output_file.write("Текст;Метка\n")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Обработка данных"):
            заголовок = str(row["Заголовок"]) if pd.notna(row["Заголовок"]) else ""
            сообщение = str(row["Сообщение"]) if pd.notna(row["Сообщение"]) else ""
            text = f"{заголовок} {сообщение}".strip()
            logging.debug(f"Исходный текст: {text}")

            processed_text = preprocessor.preprocess_text(text)
            logging.debug(f"Обработанный текст: {processed_text}")

            if processed_text.strip():
                output_file.write(f"{processed_text};{row['id_метки']}\n")
                output_file.flush()  # Принудительно записываем данные на диск

    logging.info(f"Сохранено обработанных данных в {PROCESSED_DATA_PATH}")

def main():
    """
    Основной метод: проверяет наличие файла raw/requests.csv,
    загружает данные и выполняет предобработку.
    """
    try:
        if os.path.exists(RAW_DATA_PATH):
            logging.info(f"Файл {RAW_DATA_PATH} найден, загружаем данные...")
            df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8", sep=";")
        else:
            df = load_data_from_db()
            if df is None:
                logging.error("Ошибка: Данных для обработки нет.")
                return

        preprocess_and_save_data(df)
    except Exception as e:
        logging.error(f"Ошибка в основном методе: {e}")

if __name__ == "__main__":
    main()