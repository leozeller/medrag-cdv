"""Tests for generation helpers (passage formatting + thinking-token stripping).

The actual LLM invocation is not tested — it requires Ollama running and
is verified manually via a smoke run.
"""

from __future__ import annotations

from src.generation import _format_passages, _strip_thinking
from src.passages import Passage


def test_strip_thinking_removes_paired_tokens():
    text = "<unused94>let me think about this...<unused95>The answer is X."
    assert _strip_thinking(text) == "The answer is X."


def test_strip_thinking_handles_unclosed_block_via_last_paragraph():
    text = "<unused94>step 1\n\nstep 2\n\nFinal answer here."
    assert _strip_thinking(text) == "Final answer here."


def test_format_passages_numbers_and_separates():
    p1 = Passage(chunk_id="a::0", doc_id="a", text="First.", score=0.9)
    p2 = Passage(chunk_id="b::0", doc_id="b", text="Second.", score=0.8)
    block = _format_passages([p1, p2])

    assert "[1] First." in block
    assert "[2] Second." in block
    assert block.index("[1]") < block.index("[2]")
