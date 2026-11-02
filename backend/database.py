import uuid
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Index, text
from sqlalchemy.orm import declarative_base, sessionmaker

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./analyses.db")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow)
    similarity_score = Column(Float)
    matched_skills = Column(JSON)
    missing_skills = Column(JSON)
    transferable_skills = Column(JSON)
    section_scores = Column(JSON)
    feedback = Column(JSON)

    __table_args__ = (
        Index("ix_analyses_timestamp_id", "timestamp", "id"),
    )


def ensure_indexes(bind):
    with bind.begin() as connection:
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_analyses_timestamp_id "
            "ON analyses (timestamp, id)"
        ))
        connection.execute(text("DROP INDEX IF EXISTS ix_analyses_id"))

# Create tables
Base.metadata.create_all(bind=engine)
ensure_indexes(engine)
