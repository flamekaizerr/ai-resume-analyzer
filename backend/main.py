from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import uuid
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from database import get_db, AnalysisRecord
from db_queries import get_history as query_history, get_stats as query_stats
from schemas import AnalyzeRequest, AnalysisResponse, SectionScores
from models import ml_models
from parser import extract_text_from_upload, segment_resume
from analyzer import compute_similarity, analyze_gap, compute_section_scores
from feedback import generate_feedback

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML models on startup
    ml_models.load_models()
    yield
    # Clean up on shutdown if necessary

app = FastAPI(title="AI Resume Screener & Job-Fit Analyzer", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangGraph AI Agent Router (additive)
try:
    from agent.router import router as agent_router
    app.include_router(agent_router, prefix="/api")
    app.include_router(agent_router, prefix="", include_in_schema=False)
except Exception as _agent_err:
    import logging
    logging.getLogger(__name__).warning("Agent router could not be loaded: %s", _agent_err)

# Question answering over ingested resume/JD files (see rag/)
try:
    from rag.router import router as rag_router
    app.include_router(rag_router, prefix="/api")
    app.include_router(rag_router, prefix="", include_in_schema=False)
except Exception as _rag_err:
    import logging
    logging.getLogger(__name__).warning("RAG router could not be loaded: %s", _rag_err)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_documents(
    jd_text: str = Form(...),
    resume_file: UploadFile = File(None),
    resume_text: str = Form(None),
    db: Session = Depends(get_db)
):
    if not resume_file and not resume_text:
        raise HTTPException(status_code=400, detail="Must provide either resume_file or resume_text")
        
    extracted_resume_text = resume_text or ""
    
    if resume_file:
        try:
            content = await resume_file.read()
            extracted_resume_text = extract_text_from_upload(resume_file.filename or "resume.txt", content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
            
    # Process
    try:
        sections = segment_resume(extracted_resume_text)
        similarity = compute_similarity(extracted_resume_text, jd_text)
        gap_data = analyze_gap(extracted_resume_text, jd_text)
        feedback_bullets = generate_feedback(gap_data["missing_skills"], similarity)

        section_scores_data = compute_section_scores(
            resume_text=extracted_resume_text,
            jd_text=jd_text,
            resume_sections=sections,
            resume_skills=gap_data["resume_skills"],
            jd_skills=gap_data["jd_skills"],
            similarity_score=similarity,
        )

        section_scores = SectionScores(**section_scores_data)
        
        record_id = uuid.uuid4()
        
        response_data = AnalysisResponse(
            id=record_id,
            timestamp=datetime.utcnow(),
            similarity_score=similarity,
            matched_skills=gap_data["matched_skills"],
            missing_skills=gap_data["missing_skills"],
            transferable_skills=gap_data["transferable_skills"],
            section_scores=section_scores,
            feedback=feedback_bullets,
            resume_text=extracted_resume_text,
        )
        
        # Save to DB
        db_record = AnalysisRecord(
            id=str(record_id),
            timestamp=response_data.timestamp,
            similarity_score=similarity,
            matched_skills=gap_data["matched_skills"],
            missing_skills=gap_data["missing_skills"],
            transferable_skills=gap_data["transferable_skills"],
            section_scores=section_scores.model_dump(),
            feedback=feedback_bullets
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        return response_data
    except Exception as analyze_error:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())

@app.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    return AnalysisResponse(
        id=uuid.UUID(record.id),
        timestamp=record.timestamp,
        similarity_score=record.similarity_score,
        matched_skills=record.matched_skills,
        missing_skills=record.missing_skills,
        transferable_skills=record.transferable_skills,
        section_scores=SectionScores(**record.section_scores),
        feedback=record.feedback
    )

@app.get("/history")
def get_history(
    limit: int = Query(None, ge=1, le=100),
    before: str = Query(None),
    min_score: float = Query(None),
    db: Session = Depends(get_db),
):
    return query_history(db, limit=limit, before=before, min_score=min_score)

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    return query_stats(db)

@app.delete("/analysis/{analysis_id}")
def delete_analysis(analysis_id: str, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    db.delete(record)
    db.commit()
    return {"status": "deleted"}
