import os

PG_PASSWORD = os.getenv('POSTGRES_PASSWORD')
PG_USER = os.getenv('POSTGRES_USER')
PG_DB = os.getenv('POSTGRES_DB')
PG_HOST = os.getenv('POSTGRES_HOST')
PG_PORT = os.getenv('POSTGRES_PORT', '5432')

PG_DSN = f'postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}'