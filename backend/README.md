---
title: AI Resume Analyzer API
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AI Resume Analyzer API

FastAPI backend for resume-to-job-description analysis, skill gap detection, section scoring, and feedback generation.

## Endpoints

- `POST /analyze`
- `GET /history`
- `GET /stats`
- `GET /analysis/{analysis_id}`
- `DELETE /analysis/{analysis_id}`
- `POST /api/rag/ask` (needs an index from `python -m rag ingest`, see the main README)

The service listens on port `7860` when deployed as a Hugging Face Docker Space.
