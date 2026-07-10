# ============================================================================
# StormSentinel Backend — Database Setup
# SQLAlchemy engine + session management. get_db() is a FastAPI dependency
# that yields a session and always closes it, even if the request errors.
# ============================================================================

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
