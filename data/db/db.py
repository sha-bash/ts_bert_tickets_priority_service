from typing import Optional
import os
import pandas as pd
import logging
import pyodbc
import yaml

# Настройка логирования
os.makedirs("data/logs", exist_ok=True)
logging.basicConfig(
    filename="data/logs/preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DatabaseManager:
    def __init__(self, connect_to_db_06: bool = False):
        self._connection = None
        try:
            with open("/app_confs/app_conf.yaml", 'r') as stream:
                config = yaml.safe_load(stream)
        except FileNotFoundError:
            with open("config/settings.yaml", 'r') as stream:
                config = yaml.safe_load(stream)
                config = config.get('conf')

        if connect_to_db_06:
            self.driver: str = config['driver']
            self.host: str = config['SERVER_MSSM_06']
            self.port: str = config['PORT_MSSM_06']
            self.db_name: str = config['DATABASE_MSSM_06']
            self.db_user: str = config['UID_MSSM_06']
            self.db_password: str = config['PWD_MSSM_06']
        else:
            self.driver: str = config['driver']
            self.host: str = config['SERVER_MSSM']
            self.port: str = config['PORT_MSSM']
            self.db_name: str = config['DATABASE_MSSM']
            self.db_user: str = config['UID_MSSM']
            self.db_password: str = config['PWD_MSSM']

    def __enter__(self):
        connection_params = {
            'Driver': self.driver,
            'Server': f'{self.host},{self.port}',
            'Database': self.db_name,
            'UID': self.db_user,
            'PWD': self.db_password,
        }
        connection_string = ';'.join(
            [f'{key}={value}' for key, value in connection_params.items()] + ['TrustServerCertificate=yes']
        )
        self._connection = pyodbc.connect(connection_string)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._connection:
            self._connection.close()

    def exec_query(self, command: str, *args: tuple, fetchval: bool = False, fetchall: bool = False, fetchone: bool = False, response_as_dict: bool = True) -> Optional[any]:
        with self._connection.cursor() as cursor:
            cursor.execute(command, *args)

            fetch_actions = {
                fetchval: cursor.fetchval,
                fetchall: cursor.fetchall,
                fetchone: cursor.fetchone
            }
            result_action = fetch_actions.get(True)
            result = result_action() if result_action else None

            if response_as_dict and (fetchone or fetchall):
                columns_name = [column[0] for column in cursor.description]
                result = self.__transform_response_to_dict(result, columns_name, fetchone_mode=fetchone)

        return result

    @staticmethod
    def __transform_response_to_dict(rows: list, columns_name: list, fetchone_mode: bool) -> list:
        responses = []
        wrapper_list = []
        if fetchone_mode:
            wrapper_list.append(rows)
            rows = wrapper_list

        for row in rows:
            row_dict = dict(zip(columns_name, row))
            responses.append(row_dict)

        return responses

    def get_requests_data(self):
        request_data_query = '''
            WITH FirstMessage AS (
            SELECT 
                d.number_mistake,
                d.text_message,
                d.created_at,
                ROW_NUMBER() OVER (PARTITION BY d.number_mistake ORDER BY d.created_at) AS rn
            FROM Bot2l.dbo.[description] d (NOLOCK)
            )
            SELECT 
            m.mistake_name AS 'Заголовок',
            ISNULL(fm.text_message, 'Нет данных') AS 'Сообщение',
            ll.mistake_label_id AS 'id_метки',
            ml.[name] AS 'Название_команды',
            ml.command_number_2l AS 'Номер_команды'
            FROM Bot2l.dbo.mistakes m (NOLOCK)
            INNER JOIN FirstMessage fm 
                ON fm.number_mistake = m.id
            INNER JOIN Bot2l.dbo.[labels-lists] ll (NOLOCK) 
                ON ll.mistake_id = m.id
            INNER JOIN [Bot2l].[dbo].[mistake-labels] ml (NOLOCK) 
                ON ml.id = ll.mistake_label_id
            WHERE 
                fm.rn IN (2)
                AND ll.mistake_label_id NOT IN (1,4,5,6,19,21,25,29,30,43,44,45,46,47);
            '''
        return self.exec_query(request_data_query, fetchall=True)