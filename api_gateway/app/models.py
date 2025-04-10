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
    text_task: Mapped[str] = mapped_column(String, nullable=False)
    task_priority: Mapped[str] = mapped_column(String, nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False)
    date_add: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    @property
    def dict(self):
        return {
            'id': self.id,
            'text_task': self.text_task,
            'task_priority': self.task_priority,
            'priority_score': self.priority_score,
            'date_add': self.date_add.isoformat(),
        }
    

ORM_OBJECT = Prediction
ORM_CLS = type[Prediction]