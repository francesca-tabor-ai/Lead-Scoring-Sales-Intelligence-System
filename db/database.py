"""
Database engine, session management, and initialization.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

from config.settings import settings
from db.models import Base

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized.")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_dependency() -> Generator[Session, None, None]:
    """FastAPI dependency injector."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
