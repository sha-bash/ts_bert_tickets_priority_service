from config import PG_DSN
from sqlalchemy.ext.asyncio import (AsyncAttrs, async_sessionmaker, create_async_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Float, func, DateTime
import datetime

engine = create_async_engine(
    PG_DSN,
)

Session = async_sessionmaker(bind=engine, expire_on_commit=False)

class Base(AsyncAttrs, DeclarativeBase):

    @property
    def id_dict(self):
        return {"id": self.id}

class Prediction(Base):

    __tablename__ = 'prediction'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    custom_task_priority: Mapped[str] = mapped_column(String, nullable=False)
    custom_task_score: Mapped[float] = mapped_column(Float, nullable=False)
    task_priority: Mapped[str] = mapped_column(String, nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False)
    date_add: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    @property
    def dict(self):
        return {
            'id': self.id,
            'custom_task_priority': self.custom_task_priority,
            'custom_priority_score': self.custom_task_score,
            'task_priority': self.task_priority,
            'priority_score': self.priority_score,
            'date_add': self.date_add.isoformat(),
        }
    

ORM_OBJECT = Prediction
ORM_CLS = type[Prediction]