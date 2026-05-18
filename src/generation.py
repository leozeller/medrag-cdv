"""LLM-based answer generation over retrieved passages.

Uses Ollama via `langchain-ollama`. Model and host are configurable via
the `OLLAMA_MODEL` and `OLLAMA_HOST` env vars. The prompt template is
locked for the duration of the sprint (see decision T8 in
`plan/decisions.md`) so retriever comparisons are not confounded by
prompt-engineering effects.
"""

from __future__ import annotations

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.passages import Passage

DEFAULT_MODEL = "medgemma:4b"
DEFAULT_HOST = "http://localhost:11434"
# 2048 matches Ollama's own default num_ctx — keeping it identical avoids
# triggering a model reload on first request, which can stall for minutes.
DEFAULT_NUM_CTX = 2048

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical question answering assistant. Answer the question "
            "based on the provided passages. Use only information from the "
            "passages — do not add facts from your own training. If the passages "
            "do not contain enough information, say so. Keep the answer concise "
            "and factual.",
        ),
        (
            "user",
            "Passages:\n{passages_block}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)

# Prompt for the LLM-only baseline (no retrieval). The model answers from
# its own knowledge. Used to isolate the contribution of retrieval in eval.
NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical question answering assistant. Answer the "
            "question concisely and factually from your own knowledge. If "
            "you don't know the answer with confidence, say \"I don't know\" "
            "rather than guess.",
        ),
        ("user", "Question: {question}\n\nAnswer:"),
    ]
)


def _format_passages(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))


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
            # Hard cap on generated tokens — prevents the occasional runaway
            # generation (e.g. on open-ended treatment questions where the
            # model lists indefinitely). 512 tokens ≈ ~50 sec on CPU.
            num_predict=512,
            # 8 threads passed per request — Ollama doesn't have a global
            # env var for this; threads are a per-request llama.cpp option.
            num_thread=8,
            # Keep the model warm in Ollama between calls so we don't pay
            # the 30–60 sec reload cost on cold start. Default 1h — set
            # OLLAMA_KEEP_ALIVE=-1 to keep it indefinitely (e.g. for long
            # overnight runs across multiple retrievers).
            keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "1h"),
            # Non-streaming: wait for the full response in one shot.
            disable_streaming=True,
        )
        self._chain = PROMPT | self._llm
        self._chain_no_context = NO_CONTEXT_PROMPT | self._llm

    def generate(self, question: str, passages: list[Passage]) -> str:
        if passages:
            response = self._chain.invoke(
                {
                    "question": question,
                    "passages_block": _format_passages(passages),
                }
            )
        else:
            # LLM-only baseline — answer from internal knowledge only.
            response = self._chain_no_context.invoke({"question": question})
        content = response.content
        if isinstance(content, list):
            content = "".join(
                part if isinstance(part, str) else str(part.get("text", ""))
                for part in content
            )
        return content
