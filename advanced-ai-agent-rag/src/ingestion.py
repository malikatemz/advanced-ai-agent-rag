"""
Document ingestion: load raw files (pdf, txt, md, docx) and split them into
overlapping chunks suitable for embedding.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument

from src.config import get_settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    doc_id: str
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.doc_id}-{self.chunk_index}"


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"[page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext in (".txt", ".md"):
        return _read_text(path)
    raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter - avoids pulling in a heavy NLP dependency.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    source: str,
    doc_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks, trying to break on sentence
    boundaries so chunks stay semantically coherent.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    text = _clean_text(text)
    if not text:
        return []

    sentences = _split_into_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    def flush():
        nonlocal current, current_len, idx
        if not current:
            return
        chunk_str = " ".join(current).strip()
        if chunk_str:
            chunks.append(Chunk(text=chunk_str, source=source, chunk_index=idx, doc_id=doc_id))
            idx += 1

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > chunk_size and current:
            flush()
            # carry overlap forward (last N chars worth of sentences)
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += sentence_len

    flush()

    # Fallback: if a single "sentence" is longer than chunk_size (e.g. no
    # punctuation at all), hard-split it by characters.
    final_chunks: list[Chunk] = []
    next_idx = 0
    for c in chunks:
        if len(c.text) <= chunk_size * 1.5:
            c.chunk_index = next_idx
            final_chunks.append(c)
            next_idx += 1
        else:
            for i in range(0, len(c.text), chunk_size - chunk_overlap):
                piece = c.text[i : i + chunk_size]
                if piece.strip():
                    final_chunks.append(
                        Chunk(text=piece.strip(), source=source, chunk_index=next_idx, doc_id=doc_id)
                    )
                    next_idx += 1

    return final_chunks


def make_doc_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    return f"{path.stem}-{digest}"


def ingest_file(path: str | Path) -> list[Chunk]:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    text = load_document_text(path)
    doc_id = make_doc_id(path)
    return chunk_text(text, source=str(path.name), doc_id=doc_id)


def ingest_directory(directory: str | Path) -> list[Chunk]:
    directory = Path(directory)
    all_chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_chunks.extend(ingest_file(path))
    return all_chunks
