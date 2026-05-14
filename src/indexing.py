"""Build a FAISS dense retrieval index over MedQuAD's Disorders subset.

Uses the shared chunk pool from `src.chunking` so that the dense and
BM25 retrievers compare on identical chunks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    prepare_chunks,
)

DEFAULT_INPUT = Path("data/processed/medquad.jsonl")
DEFAULT_OUTPUT = Path("results/indices/dense")
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def build_index(
    input_path: Path,
    output_path: Path,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    print(f"Preparing chunks from {input_path}…")
    texts, metadatas = prepare_chunks(input_path, chunk_size, chunk_overlap)
    n_docs = len({m["doc_id"] for m in metadatas})
    print(
        f"  {len(texts)} chunks from {n_docs} docs "
        f"(avg {len(texts) / n_docs:.1f}/doc)"
    )

    print(f"Loading embedding model: {model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        encode_kwargs={"normalize_embeddings": True, "show_progress_bar": True},
    )

    print("Embedding chunks + building FAISS index…")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    output_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_path))
    print(f"Index saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP
    )
    args = parser.parse_args()
    build_index(
        args.input,
        args.output,
        args.model,
        args.chunk_size,
        args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
