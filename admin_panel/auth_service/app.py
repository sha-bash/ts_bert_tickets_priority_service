import os
import jwt
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from werkzeug.security import generate_password_hash, check_password_hash

from models import Session, AdminUser
from config import PG_DSN

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = PG_DSN
app.config["SECRET_KEY"] = os.getenv('FLASK_SECRET_KEY')


async def get_user_by_username(session: AsyncSession, username: str):
    """Асинхронный поиск пользователя по имени"""
    result = await session.execute(select(AdminUser).where(AdminUser.username == username))
    return result.scalars().first()


@app.route("/register", methods=["POST"])
async def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    async with Session() as session:
        existing_user = await get_user_by_username(session, username)
        if existing_user:
            return jsonify({"error": "User already exists"}), 400

        user = AdminUser(
            username=username,
            password_hash=generate_password_hash(password),
            role="moderator"
        )

        session.add(user)
        await session.commit()

        return jsonify({"message": "User created"}), 201


@app.route("/login", methods=["POST"])
async def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    async with Session() as session:
        user = await get_user_by_username(session, username)
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid credentials"}), 401

        token = jwt.encode(
            {
                "user_id": user.id,
                "role": user.role,
                "exp": datetime.utcnow() + timedelta(hours=1)
            },
            app.config["SECRET_KEY"],
            algorithm="HS256"
        )
        return jsonify({"token": token})


if __name__ == "__main__":
    import asyncio
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:8000"]
    asyncio.run(serve(app, config))
