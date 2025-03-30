from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select
from typing import Type
from werkzeug.exceptions import HTTPException
from models import AdminUser


async def add_user(session: AsyncSession, item: AdminUser) -> AdminUser:
    """Добавляет нового пользователя"""
    session.add(item)
    try:
        await session.commit()
        return item
    except IntegrityError as e:
        await session.rollback()  
        if "23505" in str(e.orig):  # Код ошибки уникального ограничения PostgreSQL
            raise HTTPException(409, "Username already exists")
        raise e


async def get_user(session: AsyncSession, orm_class: Type[AdminUser], item_id: int) -> AdminUser:
    """Получает пользователя по ID"""
    orm_obj = await session.get(orm_class, item_id)
    if orm_obj is None:
        raise HTTPException(404, "User not found")
    return orm_obj


async def delete_user(session: AsyncSession, orm_class: Type[AdminUser], item_id: int):
    """Удаляет пользователя по ID"""
    orm_obj = await session.get(orm_class, item_id)
    if orm_obj is None:
        raise HTTPException(404, "User not found")
    
    await session.delete(orm_obj)
    await session.commit()
