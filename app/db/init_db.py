from sqlalchemy import text

from app.db import models  # noqa: F401
from app.db.session import Base, engine


def init_db() -> None:
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Base de datos inicializada correctamente.")
