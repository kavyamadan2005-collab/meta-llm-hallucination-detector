"""SQLAlchemy engine and session factory."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.core.config import get_settings


settings = get_settings()
database_path = Path(settings.database_path)
if not database_path.is_absolute():
    database_path = Path.cwd() / database_path

database_path.parent.mkdir(parents=True, exist_ok=True)
database_url = f"sqlite:///{database_path.as_posix()}"

engine = create_engine(
    database_url,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Iterator[Session]:
    """Yield a database session and close it after use."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
