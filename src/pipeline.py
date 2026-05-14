"""End-to-end retrieval + generation pipeline for medrag-cdv.

Loads MedQuAD records, samples a deterministic test set from the
Disorders subset, runs the chosen retriever, generates answers with the
LLM, and writes per-question results to `results/runs/{retriever}.jsonl`.

The output file is *append-only* and *resumable*: re-running with the
same `--retriever` skips qids already present in the output.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from src.generation import Generator
from src.passages import Passage
from src.retrievers import Retriever
from src.retrievers.dense import DenseRetriever

DEFAULT_INPUT = Path("data/processed/medquad.jsonl")
DEFAULT_INDEX = Path("results/indices/dense")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_OUTPUT_DIR = Path("results/runs")
DEFAULT_SAMPLE_SIZE = 300
DEFAULT_SEED = 42
DEFAULT_K = 5


def load_sample(path: Path, sample_size: int, seed: int) -> list[dict]:
    """Load Disorders records and sample `sample_size` deterministically."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("semantic_group") == "Disorders":
                records.append(r)
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:sample_size]


def already_done_qids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["qid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _passage_to_dict(p: Passage) -> dict:
    return {"chunk_id": p.chunk_id, "doc_id": p.doc_id, "score": p.score}


def _prepare_records(
    input_path: Path,
    sample_size: int,
    seed: int,
    limit: int | None,
    resume: bool,
    output_path: Path,
) -> list[dict]:
    print("Loading sample…")
    records = load_sample(input_path, sample_size, seed)
    if limit is not None:
        records = records[:limit]
    print(f"  {len(records)} records to process")

    done = already_done_qids(output_path) if resume else set()
    if done:
        print(f"  resuming, skipping {len(done)} already-done qids")
    return [r for r in records if r["qid"] not in done]


def _run_loop(
    retriever: Retriever,
    records: list[dict],
    generator: Generator,
    output_path: Path,
    k: int,
    desc: str,
) -> None:
    with output_path.open("a", encoding="utf-8") as fh:
        for record in tqdm(records, desc=desc):
            passages = retriever.retrieve(record["question"], k=k)
            answer = generator.generate(record["question"], passages)
            output = {
                "qid": record["qid"],
                "doc_id": record["doc_id"],
                "question": record["question"],
                "gold_answer": record["answer"],
                "passages": [_passage_to_dict(p) for p in passages],
                "generated_answer": answer,
            }
            fh.write(json.dumps(output, ensure_ascii=False))
            fh.write("\n")
            fh.flush()


def run(
    retriever_name: str,
    input_path: Path,
    output_path: Path,
    *,
    index_path: Path,
    embedding_model: str,
    sample_size: int,
    seed: int,
    k: int,
    limit: int | None,
    resume: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _prepare_records(
        input_path, sample_size, seed, limit, resume, output_path
    )
    if not records:
        print("Nothing to do.")
        return

    print(f"Loading FAISS index from {index_path}…")
    retriever: Retriever = DenseRetriever(index_path, embedding_model)

    print("Loading LLM…")
    generator = Generator()

    print(f"Running pipeline on {len(records)} questions…")
    _run_loop(retriever, records, generator, output_path, k, desc=retriever_name)
    print(f"Done. Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retriever",
        choices=["dense"],
        required=True,
        help="Which retriever to run. (bm25/entity-aware come in later phases)",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: results/runs/{retriever}.jsonl).",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions actually processed (smoke testing).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already-processed qids in the output file.",
    )
    args = parser.parse_args()

    output = args.output or (DEFAULT_OUTPUT_DIR / f"{args.retriever}.jsonl")
    run(
        retriever_name=args.retriever,
        input_path=args.input,
        output_path=output,
        index_path=args.index,
        embedding_model=args.embedding_model,
        sample_size=args.sample_size,
        seed=args.seed,
        k=args.k,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
