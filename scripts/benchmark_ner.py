import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from analyzer import extract_skills
from models import ml_models

if __name__ == "__main__":
    ml_models.load_models()
    text = "Experience with Python, Django, Docker and React."
    print("Extracted:", extract_skills(text))
