"""BM25 retriever over the same chunk pool as the dense retriever.

Builds the BM25 index in memory at construction time. With ~24k chunks
this takes a few seconds, which is acceptable for a one-shot pipeline
run. The chunk pool is reproduced via `prepare_chunks`, so dense and
BM25 share identical chunks — the comparison is apples-to-apples.
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    prepare_chunks,
)
from src.passages import Passage

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric word tokens. Same scheme used for query + docs."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(
        self,
        input_path: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._texts, self._metadatas = prepare_chunks(
            input_path, chunk_size, chunk_overlap
        )
        tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[Passage]:
        scores = self._bm25.get_scores(_tokenize(query))
        top_k = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [
            Passage(
                chunk_id=self._metadatas[i]["chunk_id"],
                doc_id=self._metadatas[i]["doc_id"],
                text=self._texts[i],
                score=float(scores[i]),
            )
            for i in top_k
        ]
