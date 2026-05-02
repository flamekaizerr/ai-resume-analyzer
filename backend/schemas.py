from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID

class SectionScores(BaseModel):
    technical_skills: float = 0.0
    soft_skills: float = 0.0
    experience_match: float = 0.0
    education: float = 0.0
    keywords: float = 0.0
    overall: float = 0.0

class AnalysisResponse(BaseModel):
    id: UUID
    timestamp: datetime
    similarity_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    transferable_skills: List[str]
    section_scores: SectionScores
    feedback: List[str]

class AnalyzeRequest(BaseModel):
    resume_text: Optional[str] = None
    jd_text: str
