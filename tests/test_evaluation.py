"""Tests for evaluation: minimal essential coverage.

Three tests:
1. ROUGE-L defensive zero-return for empty input (our defensive code).
2. aggregate() defensive zero-return for empty list (our defensive code).
3. End-to-end JSONL → per-question metrics → aggregate. Fixture data is
   crafted so the assertions implicitly verify recall@k, MRR, ROUGE-L,
   AND chunk-level dedup.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation import (
    aggregate,
    evaluate_run,
    is_abstention,
    passage_overlap,
    rouge_l_f1,
)


def test_rouge_l_f1_handles_empty():
    assert rouge_l_f1("", "anything") == 0.0
    assert rouge_l_f1("anything", "") == 0.0


def test_aggregate_handles_empty():
    assert aggregate([]) == {
        "n": 0,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "rouge_l_f1": 0.0,
        "bertscore_f1": 0.0,
        "abstention_rate": 0.0,
        "passage_overlap": 0.0,
    }


def test_is_abstention_detects_phrases():
    # Triggers
    assert is_abstention("I don't know the answer.") is True
    assert is_abstention("Not enough information in the provided passages.") is True
    assert is_abstention("The passages do not contain enough info.") is True
    # Doesn't trigger
    assert is_abstention("Hypertension is high blood pressure.") is False
    assert is_abstention("") is False


def test_passage_overlap_grounded_vs_fabricated():
    passages = ["Hypertension is high blood pressure."]
    # Fully grounded — every word also in passage
    assert passage_overlap("Hypertension is high blood pressure", passages) == 1.0
    # Partially grounded
    grounded = passage_overlap("Hypertension is a chronic condition", passages)
    assert 0 < grounded < 1
    # Empty passage list (LLM-only baseline) → 0.0
    assert passage_overlap("Hypertension is high blood pressure", []) == 0.0
    # Empty answer → 0.0
    assert passage_overlap("", passages) == 0.0


def test_evaluate_run_and_aggregate_end_to_end(tmp_path: Path):
    """Full pipeline: JSONL → per-question metrics → aggregate.

    Q1 (gold D1): chunks ordered [D2, D2, D1] — D1 sits at *unique* rank 2,
      so MRR=0.5 requires chunk-dedup to be working. Gold answer matches
      exactly → high ROUGE-L.
    Q2 (gold D3): chunks have no D3 → recall=0, MRR=0, garbage generation
      → low ROUGE-L.
    """
    run_path = tmp_path / "test.jsonl"
    records = [
        {
            "qid": "D1-1",
            "doc_id": "D1",
            "qtype": "information",
            "question": "What is X?",
            "gold_answer": "X is a disease.",
            "passages": [
                {"chunk_id": "D2::0", "doc_id": "D2", "score": 0.9},
                {"chunk_id": "D2::1", "doc_id": "D2", "score": 0.8},
                {"chunk_id": "D1::0", "doc_id": "D1", "score": 0.7},
            ],
            "generated_answer": "X is a disease.",
        },
        {
            "qid": "D3-1",
            "doc_id": "D3",
            "qtype": "symptoms",
            "question": "What are the symptoms?",
            "gold_answer": "Persistent cough and fever.",
            "passages": [{"chunk_id": "D2::0", "doc_id": "D2", "score": 0.5}],
            "generated_answer": "xyzzy completely unrelated",
        },
    ]
    with run_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    # with_bertscore=False to avoid downloading a 440 MB BERT model
    # just to satisfy a unit test; the bert-score wrapper is covered
    # implicitly by being used in production.
    per_q = evaluate_run(run_path, k=5, with_bertscore=False)

    # Q1: D1 found at unique-rank 2 after dedup, exact gold match
    assert per_q[0]["recall_at_k"] == 1
    assert per_q[0]["mrr"] == pytest.approx(0.5)
    assert per_q[0]["rouge_l_f1"] > 0.99

    # Q2: no D3 retrieved, garbage answer
    assert per_q[1]["recall_at_k"] == 0
    assert per_q[1]["mrr"] == 0.0
    assert per_q[1]["rouge_l_f1"] < 0.2

    # Aggregate over both questions
    agg = aggregate(per_q)
    assert agg["n"] == 2
    assert agg["recall_at_k"] == 0.5
    assert agg["mrr"] == pytest.approx(0.25)
    assert agg["rouge_l_f1"] > 0.4
    assert "abstention_rate" in agg
    assert "passage_overlap" in agg
