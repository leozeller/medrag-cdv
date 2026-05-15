"""Evaluation: retrieval metrics + answer-quality metric over pipeline outputs.

Reads `results/runs/{retriever}.jsonl` files (one JSON record per question
produced by `src.pipeline`) and computes:

- **Recall@k** at doc-level: did the gold `doc_id` appear among the unique
  doc_ids of the top-k retrieved chunks?
- **MRR**: reciprocal rank of the first retrieved chunk whose `doc_id`
  matches the gold doc.
- **ROUGE-L F1**: lexical overlap between the generated answer and the
  gold answer.

Aggregated metrics are written to `results/metrics.csv`, with optional
stratification by `qtype` (`--by-qtype`).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rouge_score import rouge_scorer

DEFAULT_K = 3   # matches pipeline default-k (k=3 retrieved per question)
DEFAULT_METRICS_OUTPUT = Path("results/metrics.csv")

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _unique_in_order(items: list[str]) -> list[str]:
    """Like `dict.fromkeys` but explicit; preserves first-seen order."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for item in items:
        if item not in seen_set:
            seen.append(item)
            seen_set.add(item)
    return seen


def recall_at_k(retrieved_doc_ids: list[str], gold_doc_id: str, k: int) -> int:
    """1 if `gold_doc_id` is among the first-`k` unique retrieved doc_ids, else 0."""
    return int(gold_doc_id in _unique_in_order(retrieved_doc_ids)[:k])


def mrr(retrieved_doc_ids: list[str], gold_doc_id: str) -> float:
    """1 / rank of the first retrieved doc_id matching gold, or 0 if no match."""
    for rank, doc_id in enumerate(_unique_in_order(retrieved_doc_ids), start=1):
        if doc_id == gold_doc_id:
            return 1.0 / rank
    return 0.0


def rouge_l_f1(generated: str, gold: str) -> float:
    """ROUGE-L F1 between generated and gold. Returns 0.0 on empty input."""
    if not generated or not gold:
        return 0.0
    return _ROUGE.score(gold, generated)["rougeL"].fmeasure


def evaluate_run(run_path: Path, k: int) -> list[dict]:
    """Compute per-question metrics from one pipeline output JSONL."""
    records: list[dict] = []
    with run_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            retrieved_doc_ids = [p["doc_id"] for p in r["passages"]]
            records.append(
                {
                    "qid": r["qid"],
                    "qtype": r.get("qtype", ""),
                    "recall_at_k": recall_at_k(retrieved_doc_ids, r["doc_id"], k),
                    "mrr": mrr(retrieved_doc_ids, r["doc_id"]),
                    "rouge_l_f1": rouge_l_f1(
                        r["generated_answer"], r["gold_answer"]
                    ),
                }
            )
    return records


def aggregate(per_q: list[dict]) -> dict:
    if not per_q:
        return {"n": 0, "recall_at_k": 0.0, "mrr": 0.0, "rouge_l_f1": 0.0}
    return {
        "n": len(per_q),
        "recall_at_k": mean(r["recall_at_k"] for r in per_q),
        "mrr": mean(r["mrr"] for r in per_q),
        "rouge_l_f1": mean(r["rouge_l_f1"] for r in per_q),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="One or more results/runs/{retriever}.jsonl files.",
    )
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT,
        help="Aggregated metrics CSV output.",
    )
    parser.add_argument(
        "--by-qtype",
        action="store_true",
        help="Also emit stratified metrics per qtype.",
    )
    args = parser.parse_args()

    overall_rows: list[dict] = []
    stratified_rows: list[dict] = []

    for run_path in args.runs:
        retriever_name = run_path.stem
        per_q = evaluate_run(run_path, args.k)
        overall_rows.append(
            {"retriever": retriever_name, "qtype": "all", **aggregate(per_q)}
        )

        if args.by_qtype:
            by_qtype: dict[str, list[dict]] = defaultdict(list)
            for r in per_q:
                by_qtype[r["qtype"] or "unknown"].append(r)
            for qt, recs in sorted(by_qtype.items()):
                stratified_rows.append(
                    {
                        "retriever": retriever_name,
                        "qtype": qt,
                        **aggregate(recs),
                    }
                )

    rows = overall_rows + stratified_rows
    write_csv(rows, args.output)

    for row in rows:
        print(
            f"{row['retriever']:12s} qtype={row['qtype']:18s} "
            f"n={row['n']:4d}  R@{args.k}={row['recall_at_k']:.3f}  "
            f"MRR={row['mrr']:.3f}  ROUGE-L={row['rouge_l_f1']:.3f}"
        )
    print(f"\nMetrics written to {args.output}")


if __name__ == "__main__":
    main()
