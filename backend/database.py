import uuid
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON
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

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow)
    similarity_score = Column(Float)
    matched_skills = Column(JSON)
    missing_skills = Column(JSON)
    transferable_skills = Column(JSON)
    section_scores = Column(JSON)
    feedback = Column(JSON)

# Create tables
Base.metadata.create_all(bind=engine)
