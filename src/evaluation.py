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
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rouge_score import rouge_scorer

DEFAULT_K = 3   # matches pipeline default-k (k=3 retrieved per question)
DEFAULT_METRICS_OUTPUT = Path("results/metrics.csv")

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Substrings (case-insensitive) that mark an honest abstention. These
# match either the with-passages prompt's "passages do not contain enough
# information" guidance, or the LLM-only prompt's "I don't know" fallback.
_ABSTENTION_PHRASES = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i'm unsure",
    "not enough information",
    "passages do not contain",
    "cannot answer",
    "unable to answer",
    "no information",
)


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


def bertscore_f1_batch(generated: list[str], gold: list[str]) -> list[float]:
    """Compute BertScore F1 for a list of (generated, gold) pairs.

    Lazy-imports bert-score (which downloads a ~440 MB BERT model on
    first use). Returns one F1 per pair, in the order given. Empty
    inputs are replaced with a single space so the underlying library
    doesn't NaN out — practically those will score near 0.
    """
    if not generated:
        return []
    from bert_score import score as bertscore  # lazy import

    gen_clean = [g if g.strip() else " " for g in generated]
    gold_clean = [g if g.strip() else " " for g in gold]
    _, _, f1 = bertscore(
        gen_clean,
        gold_clean,
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )
    return [float(x) for x in f1.tolist()]


def is_abstention(answer: str) -> bool:
    """True if the answer contains a known 'I don't know' / abstention phrase.

    A well-calibrated model under retrieval should abstain when passages
    don't support an answer; a poorly calibrated one hallucinates instead.
    """
    lower = answer.lower()
    return any(phrase in lower for phrase in _ABSTENTION_PHRASES)


def passage_overlap(answer: str, passage_texts: list[str]) -> float:
    """Fraction of the answer's unique tokens that also appear in any passage.

    A coarse proxy for whether the generated answer is grounded in the
    retrieved passages or fabricated. Tokens are lowercased word characters.
    Returns 0.0 when either side is empty (e.g. LLM-only baseline).
    """
    answer_tokens = {t.lower() for t in _WORD_RE.findall(answer)}
    if not answer_tokens:
        return 0.0
    passage_tokens: set[str] = set()
    for p in passage_texts:
        passage_tokens.update(t.lower() for t in _WORD_RE.findall(p))
    if not passage_tokens:
        return 0.0
    return len(answer_tokens & passage_tokens) / len(answer_tokens)


def evaluate_run(
    run_path: Path,
    k: int,
    with_bertscore: bool = True,
) -> list[dict]:
    """Compute per-question metrics from one pipeline output JSONL.

    BertScore is batched at the end because the bert-score library is
    far more efficient when given all pairs at once than when called
    once per row. Pass `with_bertscore=False` to skip (much faster, no
    BERT model download required — useful in tests).
    """
    records: list[dict] = []
    generated_list: list[str] = []
    gold_list: list[str] = []

    with run_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            retrieved_doc_ids = [p["doc_id"] for p in r["passages"]]
            # `text` is only present on newer pipeline outputs; fall back
            # to empty string so older smoke runs evaluate without error.
            passage_texts = [p.get("text", "") for p in r["passages"]]
            generated = r["generated_answer"]
            gold = r["gold_answer"]
            records.append(
                {
                    "qid": r["qid"],
                    "qtype": r.get("qtype", ""),
                    "source": r.get("source", ""),
                    "recall_at_k": recall_at_k(retrieved_doc_ids, r["doc_id"], k),
                    "mrr": mrr(retrieved_doc_ids, r["doc_id"]),
                    "rouge_l_f1": rouge_l_f1(generated, gold),
                    "bertscore_f1": 0.0,  # filled in batch below
                    "abstention": int(is_abstention(generated)),
                    "passage_overlap": passage_overlap(generated, passage_texts),
                }
            )
            generated_list.append(generated)
            gold_list.append(gold)

    if with_bertscore and records:
        f1s = bertscore_f1_batch(generated_list, gold_list)
        for record, f1 in zip(records, f1s, strict=True):
            record["bertscore_f1"] = f1

    return records


_ZERO_AGG = {
    "n": 0,
    "recall_at_k": 0.0,
    "mrr": 0.0,
    "rouge_l_f1": 0.0,
    "bertscore_f1": 0.0,
    "abstention_rate": 0.0,
    "passage_overlap": 0.0,
}


def aggregate(per_q: list[dict]) -> dict:
    if not per_q:
        return dict(_ZERO_AGG)
    return {
        "n": len(per_q),
        "recall_at_k": mean(r["recall_at_k"] for r in per_q),
        "mrr": mean(r["mrr"] for r in per_q),
        "rouge_l_f1": mean(r["rouge_l_f1"] for r in per_q),
        "bertscore_f1": mean(r["bertscore_f1"] for r in per_q),
        "abstention_rate": mean(r["abstention"] for r in per_q),
        "passage_overlap": mean(r["passage_overlap"] for r in per_q),
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
    parser.add_argument(
        "--by-source",
        action="store_true",
        help="Also emit stratified metrics per NIH source.",
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BertScore (no model download, faster eval).",
    )
    args = parser.parse_args()

    overall_rows: list[dict] = []
    stratified_rows: list[dict] = []

    for run_path in args.runs:
        retriever_name = run_path.stem
        per_q = evaluate_run(
            run_path, args.k, with_bertscore=not args.no_bertscore
        )
        overall_rows.append(
            {
                "retriever": retriever_name,
                "qtype": "all",
                "source": "all",
                **aggregate(per_q),
            }
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
                        "source": "all",
                        **aggregate(recs),
                    }
                )

        if args.by_source:
            by_source: dict[str, list[dict]] = defaultdict(list)
            for r in per_q:
                by_source[r["source"] or "unknown"].append(r)
            for src, recs in sorted(by_source.items()):
                stratified_rows.append(
                    {
                        "retriever": retriever_name,
                        "qtype": "all",
                        "source": src,
                        **aggregate(recs),
                    }
                )

    rows = overall_rows + stratified_rows
    write_csv(rows, args.output)

    show_source = args.by_source
    for row in rows:
        src_part = f" src={row['source']:18s}" if show_source else ""
        print(
            f"{row['retriever']:12s} qtype={row['qtype']:18s}{src_part} "
            f"n={row['n']:4d}  R@{args.k}={row['recall_at_k']:.3f}  "
            f"MRR={row['mrr']:.3f}  ROUGE-L={row['rouge_l_f1']:.3f}  "
            f"BertScore={row['bertscore_f1']:+.3f}  "
            f"Abstain={row['abstention_rate']:.2f}  "
            f"Overlap={row['passage_overlap']:.2f}"
        )
    print(f"\nMetrics written to {args.output}")


if __name__ == "__main__":
    main()
