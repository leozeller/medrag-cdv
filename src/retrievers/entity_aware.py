"""Entity-aware retriever: enriches the query with UMLS entities, then dense-retrieves.

The CDV-inspired variant. For each question:

1. Extract UMLS-linked medical entities via scispaCy.
2. Prepend a structured tag for each entity to the query string.
3. Pass the enriched string to the dense retriever.

The chunk index is shared with the dense retriever — only the *query*
side changes. That keeps the experimental comparison clean: any lift
over the dense baseline is attributable to entity enrichment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.passages import Passage
from src.retrievers.dense import DenseRetriever

if TYPE_CHECKING:
    from src.entity import Entity, EntityExtractor


def _enrich_query(query: str, entities: "list[Entity]") -> str:
    """Format: `[Entity: <canonical> (<cui>)] ... <original query>`.

    Empty entity list → original query unchanged.
    """
    if not entities:
        return query
    entity_block = " ".join(
        f"[Entity: {e.canonical_name} ({e.cui})]" for e in entities
    )
    return f"{entity_block} {query}"


class EntityAwareRetriever:
    def __init__(
        self,
        index_path: Path,
        embedding_model: str,
        entity_extractor: "EntityExtractor",
    ) -> None:
        self._dense = DenseRetriever(index_path, embedding_model)
        self._extractor = entity_extractor

    def retrieve(self, query: str, k: int = 5) -> list[Passage]:
        entities = self._extractor.extract(query)
        return self._dense.retrieve(_enrich_query(query, entities), k=k)
