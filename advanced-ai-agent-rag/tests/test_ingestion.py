import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import chunk_text, _clean_text, _split_into_sentences  # noqa: E402


def test_clean_text_collapses_whitespace():
    raw = "Hello   world.\r\n\r\n\r\nNext   line."
    cleaned = _clean_text(raw)
    assert "\r" not in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_split_into_sentences_basic():
    text = "This is one. This is two! Is this three?"
    sentences = _split_into_sentences(text)
    assert len(sentences) == 3


def test_chunk_text_respects_size_roughly():
    text = " ".join([f"Sentence number {i}." for i in range(200)])
    chunks = chunk_text(text, source="test.txt", doc_id="doc1", chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200 * 1.6  # allow slack for sentence boundaries
    # ids should be unique and sequential
    ids = [c.chunk_index for c in chunks]
    assert ids == list(range(len(chunks)))


def test_chunk_text_empty_input():
    assert chunk_text("   ", source="empty.txt", doc_id="doc2") == []


def test_chunk_overlap_produces_repeated_content():
    text = " ".join([f"Fact {i} about topic X." for i in range(50)])
    chunks = chunk_text(text, source="t.txt", doc_id="doc3", chunk_size=150, chunk_overlap=50)
    assert len(chunks) >= 2
    # Some overlap expected between consecutive chunks
    overlap_found = any(
        set(chunks[i].text.split()) & set(chunks[i + 1].text.split()) for i in range(len(chunks) - 1)
    )
    assert overlap_found
