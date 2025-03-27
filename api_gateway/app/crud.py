from sqlalchemy.ext.asyncio import AsyncSession
from models import ORM_OBJECT, ORM_CLS
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException


async def add_prediction(session: AsyncSession, item: ORM_OBJECT)->ORM_OBJECT:
    session.add(item)
    try:
        await session.commit()
    except IntegrityError as e:
        if e.orig.pgcode =='23505':
            raise HTTPException(status_code=409, detail='the field is not unique')
        raise e
    return item



