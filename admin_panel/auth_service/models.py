from sqlalchemy.ext.asyncio import (AsyncAttrs, async_sessionmaker, create_async_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Boolean
from config import PG_DSN
from werkzeug.security import check_password_hash


engine = create_async_engine(
    PG_DSN,
)

Session = async_sessionmaker(bind=engine, expire_on_commit=False)

class Base(AsyncAttrs, DeclarativeBase):

    @property
    def id_dict(self):
        return {"id": self.id}

class AdminUser(Base):
    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)  # Хеш пароля
    role: Mapped[str] = mapped_column(String(20), default="moderator")  # admin/moderator
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    def verify_password(self, password: str) -> bool:
        """Проверка пароля через bcrypt или аналоги."""
        return check_password_hash(self.password_hash, password)
    
    @property
    def dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'password_hash': self.password_hash,
            'role': self.role,
            'is_active': self.is_active,
        }
    

ORM_OBJECT = AdminUser
ORM_CLS = type[AdminUser]