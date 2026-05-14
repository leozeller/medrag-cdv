"""Dense retriever backed by a pre-built FAISS index."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.passages import Passage


class DenseRetriever:
    """Wraps a saved FAISS index for top-k passage retrieval.

    Returns `Passage`s with `doc_id` populated from the chunk metadata so
    the gold mapping (`qid` -> `doc_id`) in evaluation is a direct lookup.
    """

    def __init__(self, index_path: Path, model: str) -> None:
        embeddings = HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str, k: int = 5) -> list[Passage]:
        results = self._store.similarity_search_with_score(query, k=k)
        return [
            Passage(
                chunk_id=doc.metadata["chunk_id"],
                doc_id=doc.metadata["doc_id"],
                text=doc.page_content,
                score=float(score),
            )
            for doc, score in results
        ]
