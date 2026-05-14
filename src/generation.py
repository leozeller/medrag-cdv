"""LLM-based answer generation over retrieved passages.

Uses Ollama via `langchain-ollama`. Model and host are configurable via
the `OLLAMA_MODEL` and `OLLAMA_HOST` env vars. The prompt template is
locked for the duration of the sprint (see decision T8 in
`plan/decisions.md`) so retriever comparisons are not confounded by
prompt-engineering effects.

MedGemma 1.5 emits chain-of-thought wrapped in `<unused94>...<unused95>`
tokens; we strip those before returning the final answer.
"""

from __future__ import annotations

import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.passages import Passage

DEFAULT_MODEL = "medgemma1.5:4b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_NUM_CTX = 4096

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical question answering assistant. Answer the question "
            "based only on the provided passages. If the passages do not contain "
            "enough information, say so. Keep the answer concise and factual.",
        ),
        (
            "user",
            "Passages:\n{passages_block}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)

_THINKING_RE = re.compile(r"<unused\d+>.*?<unused\d+>", flags=re.DOTALL)


def _format_passages(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))


def _strip_thinking(text: str) -> str:
    """Remove MedGemma 1.5 thinking blocks.

    Handles two cases:
    1. Paired `<unused94>...<unused95>` blocks → regex-strip.
    2. Unclosed `<unused94>...` blocks → keep only the last paragraph,
       which is empirically the final answer.
    """
    cleaned = _THINKING_RE.sub("", text).strip()
    if "<unused" in cleaned:
        cleaned = cleaned.split("\n\n")[-1].strip()
    return cleaned or text.strip()


class Generator:
    """Thin wrapper around `ChatOllama` + the locked prompt template."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        self._llm = ChatOllama(
            model=model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
            base_url=host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
            temperature=0.0,
            num_ctx=num_ctx,
        )
        self._chain = PROMPT | self._llm

    def generate(self, question: str, passages: list[Passage]) -> str:
        block = _format_passages(passages)
        response = self._chain.invoke(
            {"question": question, "passages_block": block}
        )
        content = response.content
        if isinstance(content, list):
            content = "".join(
                part if isinstance(part, str) else str(part.get("text", ""))
                for part in content
            )
        return _strip_thinking(content)
