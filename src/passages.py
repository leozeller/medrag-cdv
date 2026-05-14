from pydantic import BaseModel, ConfigDict


class Passage(BaseModel):
    """A single text chunk returned by a retriever.

    `doc_id` matches the original MedQuAD document id and is what
    Recall@k is computed against. `chunk_id` uniquely identifies the
    chunk within the index.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0
