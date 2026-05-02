import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from database import engine, Base, SessionLocal, AnalysisRecord
import uuid
from datetime import datetime, timedelta
import random

Base.metadata.create_all(bind=engine)

def seed():
    db = SessionLocal()
    
    # Generate some fake past analyses
    for i in range(10):
        score = random.uniform(0.4, 0.95)
        record = AnalysisRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow() - timedelta(days=random.randint(1, 30)),
            similarity_score=score,
            matched_skills=["Python", "React", "Docker"],
            missing_skills=["Kubernetes", "AWS"] if score < 0.7 else [],
            transferable_skills=["JavaScript -> React"],
            section_scores={
                "technical_skills": score,
                "soft_skills": random.uniform(0.6, 0.9),
                "experience_match": score - 0.1,
                "education": 0.9,
                "keywords": score,
                "overall": score
            },
            feedback=["Add more metrics to your experience section."]
        )
        db.add(record)
        
    db.commit()
    print("Database seeded with 10 past analyses.")

if __name__ == "__main__":
    seed()
