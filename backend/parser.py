import io
import re
from pdfminer.high_level import extract_text

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts text from a PDF file provided as bytes."""
    with io.BytesIO(pdf_bytes) as pdf_file:
        text = extract_text(pdf_file)
    return text

def segment_resume(text: str) -> dict:
    """Segments resume text into predefined sections."""
    sections = {
        "summary": "",
        "skills": "",
        "experience": "",
        "education": ""
    }
    
    # Simple regex-based heuristics for section detection
    lines = text.split('\n')
    current_section = "summary"
    
    for line in lines:
        lower_line = line.strip().lower()
        
        if re.match(r'^(skills|technical skills|core competencies)$', lower_line):
            current_section = "skills"
            continue
        elif re.match(r'^(experience|work experience|professional experience)$', lower_line):
            current_section = "experience"
            continue
        elif re.match(r'^(education|academic background)$', lower_line):
            current_section = "education"
            continue
            
        if line.strip():
            sections[current_section] += line + "\n"
            
    return {k: v.strip() for k, v in sections.items()}
