from pydantic import BaseModel, ConfigDict, Field


class QARecord(BaseModel):
    """A single question-answer record extracted from a MedQuAD XML."""

    model_config = ConfigDict(frozen=True)

    qid: str
    doc_id: str
    source: str
    url: str
    focus: str
    cuis: list[str] = Field(default_factory=list)
    semantic_types: list[str] = Field(default_factory=list)
    semantic_group: str | None = None
    category: str | None = None
    qtype: str
    question: str
    answer: str
