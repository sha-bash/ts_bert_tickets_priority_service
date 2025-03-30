import os
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jwt

# Адреса сервисов
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:5000")
API_SERVICE_URL = os.getenv("API_SERVICE_URL", "http://localhost:8000")

# Проверка авторизации
def check_auth():
    if "token" not in st.session_state:
        token = st.text_input("Введите JWT токен:")
        if token:
            response = requests.post(f"{AUTH_SERVICE_URL}/validate_token", json={"token": token})
            if response.status_code == 200:
                st.session_state.token = token
                decoded_token = jwt.decode(token, options={"verify_signature": False})  # Без проверки подписи
                st.session_state.user = decoded_token.get("user_id", "Неизвестный")
                st.session_state.role = decoded_token.get("role", "user")
                st.rerun()
            else:
                st.error("Доступ запрещен!")
        return False
    return True

# Основной интерфейс
if check_auth():
    st.title(f"Админ-панель (User: {st.session_state.user})")

    # Вкладки
    tab1, tab2 = st.tabs(["Дашборд", "Проверка API"])

    with tab1:
        st.header("Метрики модели")
        # Запрос метрик из API
        response = requests.get(f"{API_SERVICE_URL}/metrics")
        if response.status_code == 200:
            df = pd.DataFrame(response.json())

            # Построение графика с Matplotlib и Seaborn
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=df, x="timestamp", y="accuracy", ax=ax)
            ax.set_title("Изменение Accuracy с течением времени")
            st.pyplot(fig)
        else:
            st.error("Ошибка загрузки метрик")

    with tab2:
        st.header("Ручная проверка")
        text = st.text_area("Введите текст обращения:")
        if st.button("Отправить"):
            response = requests.post(
                f"{API_SERVICE_URL}/v1/bert_prediction",
                json={"text": text},
                headers={"Authorization": f"Bearer {st.session_state.token}"}
            )
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error("Ошибка запроса")
