"""Loading files and splitting them into chunks."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int

    def to_dict(self):
        return asdict(self)


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return path.read_text(encoding="utf-8", errors="ignore")


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
