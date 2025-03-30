import os

PG_PASSWORD = os.getenv('API_POSTGRES_PASSWORD')
PG_USER = os.getenv('API_POSTGRES_USER')
PG_DB = os.getenv('API_POSTGRES_DB')
PG_HOST = os.getenv('API_POSTGRES_HOST')
PG_PORT = os.getenv('API_POSTGRES_PORT', '5432')

PG_DSN = f'postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}'