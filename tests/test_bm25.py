"""Tests for the BM25 retriever.

Builds a small in-memory index from a tiny JSONL fixture and verifies
that obvious lexical matches rank first.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.retrievers.bm25 import BM25Retriever, _tokenize


def _write_corpus(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def _record(doc_id: str, answer: str) -> dict:
    return {
        "qid": f"{doc_id}-1",
        "doc_id": doc_id,
        "source": "Test",
        "url": "https://example.org",
        "focus": f"Focus {doc_id}",
        "cuis": [],
        "semantic_types": [],
        "semantic_group": "Disorders",
        "category": None,
        "qtype": "information",
        "question": f"Question about {doc_id}",
        "answer": answer,
    }


def test_tokenize_lowercases_and_strips_punct():
    assert _tokenize("Hello, World!") == ["hello", "world"]
    assert _tokenize("HIV-1 infection") == ["hiv", "1", "infection"]


def test_bm25_prefers_lexical_match(tmp_path):
    records = [
        _record(
            "D1",
            "Lung cancer is a malignant tumor that starts in the lungs. "
            "Symptoms include persistent cough and chest pain.",
        ),
        _record(
            "D2",
            "Diabetes is a chronic condition affecting blood glucose. "
            "Symptoms include thirst and frequent urination.",
        ),
    ]
    corpus = _write_corpus(tmp_path, records)
    retriever = BM25Retriever(corpus, chunk_size=1000, chunk_overlap=0)

    results = retriever.retrieve("cough and chest pain", k=2)

    assert len(results) == 2
    assert results[0].doc_id == "D1"
    assert results[0].score > results[1].score


