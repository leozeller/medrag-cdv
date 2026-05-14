"""Tests for indexing logic (doc grouping + chunking).

The actual FAISS build is not tested here — it requires the embedding
model and is verified manually via a smoke run on real data.
"""

from __future__ import annotations

from src.chunking import chunk_documents, group_answers_by_doc


def test_group_answers_by_doc_concatenates_per_doc():
    records = [
        {"doc_id": "D1", "answer": "First answer.", "focus": "X",
         "source": "S", "url": "u"},
        {"doc_id": "D1", "answer": "Second answer.", "focus": "X",
         "source": "S", "url": "u"},
        {"doc_id": "D2", "answer": "Other answer.", "focus": "Y",
         "source": "T", "url": "v"},
    ]
    docs = group_answers_by_doc(records)

    assert set(docs) == {"D1", "D2"}
    assert "First answer." in docs["D1"]["text"]
    assert "Second answer." in docs["D1"]["text"]
    assert docs["D1"]["text"].count("answer") == 2
    assert docs["D2"]["focus"] == "Y"
    assert docs["D2"]["source"] == "T"


def test_group_answers_preserves_first_metadata():
    records = [
        {"doc_id": "D1", "answer": "a", "focus": "FocusA",
         "source": "S1", "url": "u1"},
        {"doc_id": "D1", "answer": "b", "focus": "FocusB",
         "source": "S2", "url": "u2"},
    ]
    docs = group_answers_by_doc(records)
    assert docs["D1"]["focus"] == "FocusA"
    assert docs["D1"]["source"] == "S1"


def test_chunk_documents_creates_chunk_ids_and_metadata():
    docs = {
        "D1": {
            "doc_id": "D1",
            "focus": "X",
            "source": "S",
            "url": "u",
            "text": "lorem ipsum " * 300,  # ~3600 chars
        }
    }
    texts, metadatas = chunk_documents(docs, chunk_size=600, chunk_overlap=100)

    assert len(texts) >= 2
    assert len(texts) == len(metadatas)
    for i, m in enumerate(metadatas):
        assert m["chunk_id"] == f"D1::{i}"
        assert m["doc_id"] == "D1"
        assert m["focus"] == "X"
        assert m["source"] == "S"


def test_chunk_documents_handles_multiple_docs():
    docs = {
        "D1": {"doc_id": "D1", "focus": "X", "source": "S", "url": "u",
               "text": "short text 1"},
        "D2": {"doc_id": "D2", "focus": "Y", "source": "T", "url": "v",
               "text": "short text 2"},
    }
    texts, metadatas = chunk_documents(docs, chunk_size=1200, chunk_overlap=200)

    assert len(texts) == 2
    doc_ids = {m["doc_id"] for m in metadatas}
    assert doc_ids == {"D1", "D2"}
