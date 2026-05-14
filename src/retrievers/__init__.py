"""Retriever package: BM25, dense, entity-aware — same retrieve() signature."""

from typing import Protocol

from src.passages import Passage


class Retriever(Protocol):
    """Common shape every retriever in the pipeline implements."""

    def retrieve(self, query: str, k: int) -> list[Passage]: ...
