"""
MediGenius — db/session.py
SQLAlchemy engine and session factory.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import CHAT_DB_PATH, DATABASE_URL
from app.core.logging_config import logger


def _resolve_database_url(db_path: str = CHAT_DB_PATH) -> str:
    if DATABASE_URL:
        return DATABASE_URL
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    return f"sqlite:///{db_path}"


def get_engine(db_path: str = CHAT_DB_PATH):
    """Create and return a SQLAlchemy engine for SQLite or DATABASE_URL."""
    database_url = _resolve_database_url(db_path)
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    logger.debug("Database engine created for %s", database_url)
    return create_engine(database_url, connect_args=connect_args)


def get_session_factory(engine):
    """Return a sessionmaker bound to the given engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Module-level singletons
engine = get_engine()
SessionLocal = get_session_factory(engine)
