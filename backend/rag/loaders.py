"""Loading files and splitting them into chunks."""

import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int

    def to_dict(self):
        return asdict(self)


def read_bytes(name: str, data: bytes) -> str:
    """Turn file contents into plain text based on the file extension."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        import docx

        doc = docx.Document(io.BytesIO(data))
        lines = [p.text for p in doc.paragraphs]
        # resumes often keep skills in tables, so grab those too
        for table in doc.tables:
            for row in table.rows:
                lines.append(" ".join(cell.text for cell in row.cells))
        return "\n".join(lines)
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {name} (use .txt, .md, .pdf or .docx)")


def read_file(path: Path) -> str:
    return read_bytes(path.name, path.read_bytes())


def find_files(paths: Iterable[str]) -> List[Path]:
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                    files.append(f)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
        elif not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
    return files


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into pieces of about chunk_size characters.

    Neighbouring chunks share `overlap` characters so a sentence cut at the
    edge still shows up whole in one of them. Tries to break on whitespace.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # back up to the last space so words are not cut in half
            space = text.rfind(" ", start + overlap + 1, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # move forward to a word start
        if text[start - 1] != " ":
            nxt = text.find(" ", start, end)
            if nxt != -1:
                start = nxt + 1
    return [c for c in chunks if c]


def load_documents(paths: Iterable[str], chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    chunks = []
    for f in find_files(paths):
        text = read_file(f)
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(text=piece, source=f.name, chunk_id=i))
    return chunks


def chunk_documents(docs, chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    """Chunk (name, text) pairs that did not come from disk, like pasted text or uploads."""
    chunks = []
    for name, text in docs:
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(text=piece, source=name, chunk_id=i))
    return chunks
