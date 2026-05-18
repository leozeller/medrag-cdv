"""Tests for generation helpers.

The actual LLM invocation is not tested — it requires Ollama running and
is verified manually via a smoke run.
"""

from __future__ import annotations

from src.generation import _format_passages
from src.passages import Passage


def test_format_passages_numbers_and_separates():
    p1 = Passage(chunk_id="a::0", doc_id="a", text="First.", score=0.9)
    p2 = Passage(chunk_id="b::0", doc_id="b", text="Second.", score=0.8)
    block = _format_passages([p1, p2])

    assert "[1] First." in block
    assert "[2] Second." in block
    assert block.index("[1]") < block.index("[2]")
