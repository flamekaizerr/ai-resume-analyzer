import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from analyzer import compute_similarity
from models import ml_models

if __name__ == "__main__":
    ml_models.load_models()
    t1 = "I am a software engineer with Python experience."
    t2 = "Looking for a Python developer."
    print("Sim 1:", compute_similarity(t1, t2))
