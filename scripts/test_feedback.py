import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from feedback import generate_feedback
from models import ml_models

if __name__ == "__main__":
    ml_models.load_models()
    missing = ["Docker", "Kubernetes", "AWS"]
    print("Testing feedback generation for missing skills:", missing)
    feedback = generate_feedback(missing, 0.6)
    print("\nGenerated Feedback:")
    for b in feedback:
        print("-", b)
