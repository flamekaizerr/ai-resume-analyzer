# AI Resume Analyzer

A full-stack resume analysis application that compares a resume against a job description, identifies skill gaps, scores fit across key dimensions, and provides concise improvement suggestions.

## Features

- Resume and job description comparison
- Skill gap detection for matched, missing, and transferable skills
- Section-level scoring for technical skills, soft skills, experience, education, keywords, and overall fit
- Deterministic feedback suggestions for missing skills
- Analysis history and aggregate stats tracking
- Responsive React interface with dark/light theme support
- Lazy-loaded frontend panels for optimized production builds

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
