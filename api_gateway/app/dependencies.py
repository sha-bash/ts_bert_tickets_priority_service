from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from models import Session


async def get_session()->AsyncSession:
    async with Session() as session:
        return session


SessionDependency = Annotated[AsyncSession, Depends(get_session)]

