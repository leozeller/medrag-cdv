"""Shared chunk-preparation pipeline used by all retrievers.

Reads the parsed MedQuAD JSONL, filters to the Disorders subset, groups
QA records by source document (`doc_id`), concatenates each doc's
answers, and chunks the result.

Kept separate from `src/indexing.py` so that any retriever (dense, BM25,
entity-aware) can build over the *same* chunk pool — essential for the
comparison to be apples-to-apples.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def load_disorder_records(path: Path) -> list[dict]:
    """Read the parsed JSONL and return only Disorders-subset records."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("semantic_group") == "Disorders":
                records.append(r)
    return records


def group_answers_by_doc(records: list[dict]) -> dict[str, dict]:
    """Group records by `doc_id`, concatenating their answers."""
    answers_by_doc: dict[str, list[str]] = defaultdict(list)
    meta_by_doc: dict[str, dict] = {}

    for r in records:
        doc_id = r["doc_id"]
        answers_by_doc[doc_id].append(r["answer"])
        if doc_id not in meta_by_doc:
            meta_by_doc[doc_id] = {
                "doc_id": doc_id,
                "focus": r["focus"],
                "source": r["source"],
                "url": r["url"],
            }

    for doc_id, answers in answers_by_doc.items():
        meta_by_doc[doc_id]["text"] = "\n\n".join(answers)
    return meta_by_doc


def chunk_documents(
    docs: dict[str, dict],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[dict]]:
    """Chunk each doc's text. Returns parallel (texts, metadatas) lists."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts: list[str] = []
    metadatas: list[dict] = []
    for doc in docs.values():
        for i, chunk in enumerate(splitter.split_text(doc["text"])):
            texts.append(chunk)
            metadatas.append(
                {
                    "chunk_id": f"{doc['doc_id']}::{i}",
                    "doc_id": doc["doc_id"],
                    "focus": doc["focus"],
                    "source": doc["source"],
                }
            )
    return texts, metadatas


def prepare_chunks(
    input_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[str], list[dict]]:
    """Load, filter, group, and chunk. One-shot helper for retriever inits."""
    records = load_disorder_records(input_path)
    docs = group_answers_by_doc(records)
    return chunk_documents(docs, chunk_size, chunk_overlap)
