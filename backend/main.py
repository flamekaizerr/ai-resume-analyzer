from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import uuid
from datetime import datetime

from database import get_db, AnalysisRecord
from schemas import AnalyzeRequest, AnalysisResponse, SectionScores
from models import ml_models
from parser import extract_text_from_pdf_bytes, segment_resume
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
            if resume_file.filename and resume_file.filename.lower().endswith('.pdf'):
                extracted_resume_text = extract_text_from_pdf_bytes(content)
            else:
                extracted_resume_text = content.decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
            
    # Process
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
        feedback=feedback_bullets
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
def get_history(db: Session = Depends(get_db)):
    records = db.query(AnalysisRecord).order_by(AnalysisRecord.timestamp.desc()).all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "similarity_score": r.similarity_score,
            "overall_score": (r.section_scores or {}).get("overall", r.similarity_score),
            "matched_skills": r.matched_skills or [],
            "missing_skills": r.missing_skills or [],
        }
        for r in records
    ]

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    records = db.query(AnalysisRecord).order_by(AnalysisRecord.timestamp.desc()).all()
    if not records:
        return {
            "avg_score": 0.0,
            "best_score": 0.0,
            "recent_avg": 0.0,
            "total_analyses": 0,
            "top_missing_skills": [],
            "top_matched_skills": [],
            "last_analysis_at": None,
        }

    scores = [r.similarity_score for r in records]
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    recent_scores = scores[:5]
    recent_avg = sum(recent_scores) / len(recent_scores)

    missing_counts = {}
    matched_counts = {}
    for record in records:
        for skill in record.missing_skills or []:
            missing_counts[skill] = missing_counts.get(skill, 0) + 1
        for skill in record.matched_skills or []:
            matched_counts[skill] = matched_counts.get(skill, 0) + 1

    top_missing = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_matched = sorted(matched_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "avg_score": avg_score,
        "best_score": best_score,
        "recent_avg": recent_avg,
        "total_analyses": len(records),
        "top_missing_skills": [{"skill": k, "count": v} for k, v in top_missing],
        "top_matched_skills": [{"skill": k, "count": v} for k, v in top_matched],
        "last_analysis_at": records[0].timestamp.isoformat() if records[0].timestamp else None,
    }

@app.delete("/analysis/{analysis_id}")
def delete_analysis(analysis_id: str, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    db.delete(record)
    db.commit()
    return {"status": "deleted"}
