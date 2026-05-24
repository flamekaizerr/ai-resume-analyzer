# AI Resume Analyzer

**Live App:** [https://ai-resume-analyzer-three-eosin.vercel.app/](https://ai-resume-analyzer-three-eosin.vercel.app/)

I built this full-stack resume analysis application to help job seekers compare their resumes against specific job descriptions. It identifies skill gaps, scores fit across key dimensions, and provides concise improvement suggestions.

## Features

- **Smart Comparison**: Compares your resume against any job description.
- **Skill Gap Detection**: Identifies matched, missing, and transferable skills.
- **Detailed Scoring**: Section-level scoring for technical skills, soft skills, experience, education, keywords, and overall fit.
- **Actionable Feedback**: Deterministic feedback suggestions for missing skills.
- **History & Stats**: Tracks your past analyses and shows aggregate stats.
- **Modern UI**: A responsive React interface with dark/light theme support and lazy-loaded panels for optimized performance.

## Tech Stack

### Backend

- FastAPI
- SQLAlchemy
- Sentence Transformers
- Hugging Face Transformers
- PyTorch
- pdfminer.six

### Frontend

- React
- Vite
- Recharts
- Axios
- Lucide React

## Project Structure

```text
backend/       FastAPI application, models, database, analysis logic
frontend/      Vite React application
scripts/       Test, benchmark, and seed scripts
```

## Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The API will run at:

```text
http://127.0.0.1:8000
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will run at:

```text
http://127.0.0.1:5173
```

## Production Build

```bash
cd frontend
npm run build
```

Build output is generated in:

```text
frontend/dist
```

## Verification

Run backend feedback generation:

```bash
python scripts/test_feedback.py
```

With the backend running on port `8000`, test the analyze endpoint:

```bash
python scripts/test_analyze.py
```

Run the frontend production build:

```bash
cd frontend
npm run build
```

## Deployment Notes

The frontend can be deployed as a Vite static application. The FastAPI backend should be hosted separately on a Python-compatible server and exposed through a public API URL.
