"""POST /api/rag/ask, answers from the index built with `python -m rag ingest`."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import qa
from .store import FaissStore

router = APIRouter(prefix="/rag", tags=["rag"])


class AskRequest(BaseModel):
    question: str
    k: int = Field(default=4, ge=1, le=20)


@router.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    try:
        store = FaissStore.load(qa.index_dir())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    from .embeddings import get_embedder

    return qa.answer(req.question, store, get_embedder(), k=req.k)
