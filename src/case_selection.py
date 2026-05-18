"""Pick 20 qualitative cases for poster annotation, generate a notebook.

Reads `results/runs/{none,bm25,dense,entity}.jsonl`, groups by qid, and
selects:

- **Bucket A (5)**: complementary failures — BM25 hit XOR Dense hit
- **Bucket B (5)**: entity-aware lift — Entity hit + Dense miss
  (prefer `treatment` qtype)
- **Bucket C (5)**: BM25 dominance — BM25 hit + Dense + Entity miss
  (prefer `genetic changes` qtype)
- **Bucket D (5)**: LLM-only hallucination candidates — large ROUGE-L
  gap between LLM-only and Dense (Dense > LLM-only by > 0.20)

Writes `notebooks/03_qualitative_analysis.ipynb` with one markdown cell
per case, pre-rendered with question + gold + per-retriever output +
empty annotation slots.

Re-running overwrites the notebook — do not edit while the script
will re-run. Annotations live in the notebook itself after the first
generation; commit before regenerating.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import nbformat as nbf
from rouge_score import rouge_scorer

DEFAULT_RUNS_DIR = Path("results/runs")
DEFAULT_NOTEBOOK = Path("notebooks/03_qualitative_analysis.ipynb")
RETRIEVERS = ["none", "bm25", "dense", "entity"]
SEED = 0


def load_records(runs_dir: Path) -> dict[str, dict[str, dict]]:
    """Return {qid: {retriever_name: record}}."""
    records: dict[str, dict[str, dict]] = defaultdict(dict)
    for retriever in RETRIEVERS:
        path = runs_dir / f"{retriever}.jsonl"
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                records[r["qid"]][retriever] = r
    return records


def _hit(record: dict) -> bool:
    if not record["passages"]:
        return False
    return record["doc_id"] in [p["doc_id"] for p in record["passages"]]


def _complete(by_ret: dict) -> bool:
    return all(r in by_ret for r in RETRIEVERS)


def pick_bucket_a(records: dict) -> list[str]:
    """All candidates: BM25 hit XOR Dense hit. Random order."""
    qids = [
        qid
        for qid, by_ret in records.items()
        if _complete(by_ret) and _hit(by_ret["bm25"]) != _hit(by_ret["dense"])
    ]
    rng = random.Random(SEED)
    rng.shuffle(qids)
    return qids


def pick_bucket_b(records: dict) -> list[str]:
    """All candidates: Entity hit + Dense miss. `treatment` qtype first."""
    candidates: list[tuple[int, str]] = []
    for qid, by_ret in records.items():
        if not _complete(by_ret):
            continue
        if _hit(by_ret["entity"]) and not _hit(by_ret["dense"]):
            score = 1 if by_ret["dense"]["qtype"] == "treatment" else 0
            candidates.append((score, qid))
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: -x[0])
    return [qid for _, qid in candidates]


def pick_bucket_c(records: dict) -> list[str]:
    """All candidates: BM25 hit + Dense + Entity miss. `genetic changes` first."""
    candidates: list[tuple[int, str]] = []
    for qid, by_ret in records.items():
        if not _complete(by_ret):
            continue
        if (
            _hit(by_ret["bm25"])
            and not _hit(by_ret["dense"])
            and not _hit(by_ret["entity"])
        ):
            score = 1 if by_ret["dense"]["qtype"] == "genetic changes" else 0
            candidates.append((score, qid))
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: -x[0])
    return [qid for _, qid in candidates]


def pick_bucket_d(records: dict) -> list[str]:
    """All candidates with Dense ROUGE-L > LLM-only by > 0.20, sorted by gap desc."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def rouge(rec: dict) -> float:
        gen, gold = rec["generated_answer"], rec["gold_answer"]
        if not gen or not gold:
            return 0.0
        return scorer.score(gold, gen)["rougeL"].fmeasure

    candidates: list[tuple[float, str]] = []
    for qid, by_ret in records.items():
        if not _complete(by_ret):
            continue
        gap = rouge(by_ret["dense"]) - rouge(by_ret["none"])
        if gap > 0.20:
            candidates.append((gap, qid))
    candidates.sort(reverse=True)
    return [qid for _, qid in candidates]


def _truncate(text: str, max_chars: int = 500) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


_BUCKET_DESC = {
    "A": "Complementary failures (BM25 vs Dense)",
    "B": "Entity-Aware lift over Dense",
    "C": "BM25 dominance over Dense + Entity",
    "D": "LLM-only hallucination candidate",
}
_RETRIEVER_LABEL = {
    "none": "**LLM-only**",
    "bm25": "**BM25**",
    "dense": "**Dense**",
    "entity": "**Entity-Aware**",
}


def render_case(case_num: int, bucket: str, qid: str, records: dict) -> str:
    by_ret = records[qid]
    rec = by_ret["dense"]  # any retriever works for question/gold/metadata

    lines = [
        f"## Case {case_num} — Bucket {bucket}: {_BUCKET_DESC[bucket]}",
        "",
        f"**qid:** `{qid}` | **gold_doc:** `{rec['doc_id']}` | "
        f"**qtype:** `{rec['qtype']}` | **source:** `{rec['source']}`",
        "",
        "### Question",
        "",
        rec["question"],
        "",
        "### Gold answer (truncated)",
        "",
        _truncate(rec["gold_answer"]),
        "",
        "### Generated output per retriever",
        "",
    ]
    for r in RETRIEVERS:
        retrieved = by_ret[r]["passages"]
        doc_ids = [p["doc_id"] for p in retrieved]
        hit_mark = "✓" if _hit(by_ret[r]) else "✗"
        retrieved_str = ", ".join(doc_ids) if doc_ids else "(no retrieval)"
        lines += [
            f"#### {_RETRIEVER_LABEL[r]} — gold doc in top-3: {hit_mark}",
            "",
            f"_Retrieved doc_ids:_ `{retrieved_str}`",
            "",
            f"> {_truncate(by_ret[r]['generated_answer'])}",
            "",
        ]
    lines += [
        "### Annotation",
        "",
        "_Fill in below — these notes feed directly into the poster._",
        "",
        "- **Factually correct?** (per retriever): _",
        "- **Anything hallucinated?**: _",
        "- **Which retriever helps here, and why?**: _",
        "- **One-line takeaway for the poster**: _",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def build_notebook(cases: list[tuple[int, str, str]], records: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "# Qualitative Analysis — 20 cases for the poster\n\n"
            "Auto-generated by `src/case_selection.py`. Re-running the "
            "script overwrites this file — commit your annotations before "
            "regenerating.\n\n"
            "**Buckets:**\n\n"
            "- **A (5):** Complementary failures — BM25 vs Dense disagree on the gold doc.\n"
            "- **B (5):** Entity-Aware lift — Entity hits where Dense misses.\n"
            "- **C (5):** BM25 dominance — BM25 hits where both Dense and Entity miss.\n"
            "- **D (5):** LLM-only hallucination candidates — large ROUGE-L gap between LLM-only and Dense.\n\n"
            "**How to use:** read each case, look at the per-retriever outputs side by side, "
            "and fill in the Annotation block. Short notes are fine — we want to put 3–4 cases "
            "on the poster.\n\n"
            "**Split:** Leonie takes bucket A (cases 1–5), Naeem takes "
            "buckets B + C + D (cases 6–20). See `docs/handoff-naeem.md`."
        )
    )
    for case_num, bucket, qid in cases:
        nb.cells.append(nbf.v4.new_markdown_cell(render_case(case_num, bucket, qid, records)))
    return nb


def main() -> None:
    records = load_records(DEFAULT_RUNS_DIR)
    print(f"Loaded {len(records)} questions across {len(RETRIEVERS)} retrievers")

    pickers = [
        ("A", pick_bucket_a),
        ("B", pick_bucket_b),
        ("C", pick_bucket_c),
        ("D", pick_bucket_d),
    ]
    # Dedupe across buckets so each case appears only in its highest-priority
    # bucket (A first, then B, C, D). Take 5 per bucket after dedup.
    seen: set[str] = set()
    cases: list[tuple[int, str, str]] = []
    case_num = 1
    for bucket, picker in pickers:
        candidates = picker(records)
        picked: list[str] = []
        for qid in candidates:
            if qid in seen:
                continue
            picked.append(qid)
            seen.add(qid)
            if len(picked) >= 5:
                break
        print(f"  Bucket {bucket}: {len(picked)} cases → {picked}")
        for qid in picked:
            cases.append((case_num, bucket, qid))
            case_num += 1

    print(f"\nTotal {len(cases)} cases.")

    nb = build_notebook(cases, records)
    DEFAULT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(DEFAULT_NOTEBOOK))
    print(f"Notebook written to {DEFAULT_NOTEBOOK}")


if __name__ == "__main__":
    main()
