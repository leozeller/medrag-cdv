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

from src.evaluation import aggregate, evaluate_run, rouge_l_f1


def test_rouge_l_f1_handles_empty():
    assert rouge_l_f1("", "anything") == 0.0
    assert rouge_l_f1("anything", "") == 0.0


def test_aggregate_handles_empty():
    assert aggregate([]) == {
        "n": 0,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "rouge_l_f1": 0.0,
    }


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

    per_q = evaluate_run(run_path, k=5)

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
