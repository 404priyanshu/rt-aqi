"""
Database configuration and connection management.
Uses SQLite for simplicity and portability.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Database URL - SQLite for simplicity
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/aqi_data.db")

# Create engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Required for SQLite
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency to get database session.
    Yields a session and closes it after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize the database by creating all tables.
    """
    from backend.models.aqi_model import AQIReading, ModelMetrics
    Base.metadata.create_all(bind=engine)
