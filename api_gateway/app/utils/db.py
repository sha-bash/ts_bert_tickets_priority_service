from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from models import engine

async_session_factory = async_sessionmaker(engine, expire_on_commit=False)
