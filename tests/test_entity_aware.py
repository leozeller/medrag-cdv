"""Tests for the entity-aware retriever's query-enrichment helper.

The scispaCy NER + UMLS linker step itself is not tested here — it
requires the ~1 GB `en_core_sci_md` model + UMLS knowledge base, and is
verified manually via the pipeline smoke run.
"""

from __future__ import annotations

from src.entity import Entity
from src.retrievers.entity_aware import _enrich_query


def _entity(cui: str, name: str) -> Entity:
    return Entity(text="ignored", cui=cui, canonical_name=name, confidence=0.9)


def test_enrich_query_no_entities_returns_unchanged():
    assert _enrich_query("plain query", []) == "plain query"


def test_enrich_query_prepends_entity_block():
    entities = [_entity("C0017661", "IgA Glomerulonephritis")]
    result = _enrich_query("What are the symptoms?", entities)
    assert result == (
        "[Entity: IgA Glomerulonephritis (C0017661)] What are the symptoms?"
    )


def test_enrich_query_joins_multiple_entities():
    entities = [
        _entity("C0011847", "Diabetes Mellitus"),
        _entity("C0020538", "Hypertensive Disease"),
    ]
    result = _enrich_query("treatment options?", entities)
    assert "[Entity: Diabetes Mellitus (C0011847)]" in result
    assert "[Entity: Hypertensive Disease (C0020538)]" in result
    assert result.endswith("treatment options?")
