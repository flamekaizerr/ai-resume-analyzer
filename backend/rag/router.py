"""POST /rag/ask (also under /api).

Takes a question plus optional pasted text and files as multipart form data.
If documents are sent, a temporary index is built just for that request.
Otherwise it uses the index from `python -m rag ingest`, or the sample docs
if there is no saved index.
"""

from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile

from . import qa
from .loaders import SUPPORTED_EXTS, load_documents, read_bytes
from .store import FaissStore

router = APIRouter(prefix="/rag", tags=["rag"])

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_FILES = 5

_sample_store = None


def _fallback_store(embedder):
    global _sample_store
    try:
        return FaissStore.load(qa.index_dir()), "saved"
    except FileNotFoundError:
        pass
    if _sample_store is None:
        _sample_store = qa.build_store(load_documents([str(qa.SAMPLE_DIR)]), embedder)
    return _sample_store, "sample"


async def _read_uploads(files):
    docs = []
    for f in files:
        name = f.filename or "upload.txt"
        ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in SUPPORTED_EXTS:
            raise HTTPException(status_code=400, detail=f"{name}: only .txt, .md, .pdf and .docx files are supported")
        data = await f.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(status_code=400, detail=f"{name} is larger than 5 MB")
        try:
            text = read_bytes(name, data)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Could not read {name}")
        if not text.strip():
            raise HTTPException(status_code=400, detail=f"No text found in {name}")
        docs.append((name, text))
    return docs


@router.post("/ask")
async def ask(request: Request):
    # reading the form by hand, the FastAPI version we pin has trouble
    # with an optional list of files
    form = await request.form()
    question = (form.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is empty")
    try:
        k = int(form.get("k") or 4)
    except ValueError:
        raise HTTPException(status_code=400, detail="k must be a number")
    if k < 1 or k > 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")

    files = [f for f in form.getlist("files") if isinstance(f, UploadFile) and f.filename]
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Up to {MAX_FILES} files per request")

    docs = []
    resume_text = form.get("resume_text")
    jd_text = form.get("jd_text")
    if isinstance(resume_text, str) and resume_text.strip():
        docs.append(("pasted_resume", resume_text))
    if isinstance(jd_text, str) and jd_text.strip():
        docs.append(("pasted_jd", jd_text))
    docs.extend(await _read_uploads(files))

    from .embeddings import get_embedder

    embedder = get_embedder()
    if docs:
        store, index_used = qa.store_from_texts(docs, embedder), "request"
    else:
        store, index_used = _fallback_store(embedder)

    result = qa.answer(question, store, embedder, k=k)
    result["index"] = index_used
    # only check docs the user sent, the saved index and samples already have JDs
    result["warning"] = qa.jd_warning(question, [name for name, _ in docs]) if docs else None
    return result
