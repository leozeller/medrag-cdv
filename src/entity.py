"""Biomedical entity extraction via scispaCy + UMLS linker.

Loads scispaCy's `en_core_sci_md` model with the UMLS entity linker
attached. The linker returns UMLS Concept Unique Identifiers (CUIs) for
medical entities recognised in the input text. We filter by linker
confidence (default 0.85) to discard noisy matches.

The model and its UMLS knowledge base (~1 GB on first use) must be
installed separately on the user's host before this code can run; see
README for setup.
"""

from __future__ import annotations

from typing import cast

import spacy
from pydantic import BaseModel, ConfigDict

# Importing scispacy.linking registers the "scispacy_linker" component
# with spaCy; we also use the EntityLinker type for the linker reference.
from scispacy.linking import EntityLinker

DEFAULT_MODEL = "en_core_sci_md"
DEFAULT_THRESHOLD = 0.85


class Entity(BaseModel):
    """A medical entity grounded to a UMLS concept."""

    model_config = ConfigDict(frozen=True)

    text: str             # surface form as found in input
    cui: str              # UMLS Concept Unique Identifier
    canonical_name: str   # canonical UMLS name (used for query enrichment)
    confidence: float     # linker confidence score, 0.0–1.0


class EntityExtractor:
    """UMLS-linked entity extractor.

    Loading the model + UMLS knowledge base costs several seconds and
    ~1 GB of memory. Construct once and reuse for many calls.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._nlp = spacy.load(model_name)
        self._nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "threshold": threshold,
            },
        )
        self._linker = cast(EntityLinker, self._nlp.get_pipe("scispacy_linker"))
        self._threshold = threshold

    def extract(self, text: str) -> list[Entity]:
        doc = self._nlp(text)
        entities: list[Entity] = []
        for ent in doc.ents:
            if not ent._.kb_ents:
                continue
            top_cui, top_score = ent._.kb_ents[0]
            if top_score < self._threshold:
                continue
            kb_entity = self._linker.kb.cui_to_entity[top_cui]
            entities.append(
                Entity(
                    text=ent.text,
                    cui=top_cui,
                    canonical_name=kb_entity.canonical_name,
                    confidence=float(top_score),
                )
            )
        return entities
