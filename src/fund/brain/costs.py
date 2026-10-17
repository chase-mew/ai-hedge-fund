# pyright: strict
"""Token and web search pricing for one Claude response."""

from __future__ import annotations

from typing import Protocol

from fund.config import ModelSpec


class TokenUsage(Protocol):
    """Token counts billed separately for input, output, cache write, and cache read."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


def usage_cost(usage: TokenUsage, model: ModelSpec, web_searches: int, search_cost_usd: float) -> float:
    """Price token counts at the model rates and add web search fees."""
    token_usd = (
        usage.input_tokens * model.input_per_mtok
        + usage.output_tokens * model.output_per_mtok
        + usage.cache_creation_input_tokens * model.cache_write_per_mtok
        + usage.cache_read_input_tokens * model.cache_read_per_mtok
    )
    return token_usd / 1_000_000 + web_searches * search_cost_usd
