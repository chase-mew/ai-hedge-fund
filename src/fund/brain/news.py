# pyright: strict
"""News digest. A failed search leaves the cycle free to continue without one."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

from fund.brain.llm import Llm
from fund.brain.prompts import SYSTEM_NEWS
from fund.config import ModelSpec, NewsSettings
from fund.contracts import LlmUsage, MarketSpec, NewsDigest

logger = logging.getLogger(__name__)


async def compile_news(
    claude: Llm,
    model: ModelSpec,
    settings: NewsSettings,
    universe: list[MarketSpec],
    now: datetime,
) -> tuple[NewsDigest | None, LlmUsage | None, dict[str, Any] | None]:
    """Compile a news digest. Failure returns no digest so the cycle can continue."""
    if not settings.enabled:
        return None, None, None
    result = await claude.search(
        purpose="news",
        model=model,
        system=SYSTEM_NEWS,
        user=_user_message(universe, now),
        news=settings,
    )
    if result.error is not None or result.output is None or not result.output.strip():
        logger.warning("News digest unavailable: %s", result.error or "empty digest")
        return None, result.usage, result.transcript
    digest = NewsDigest(text=result.output, sources=_sources(result.transcript), compiled_at=now)
    logger.info("News digest compiled with %s sources", len(digest.sources))
    return digest, result.usage, result.transcript


def _user_message(universe: list[MarketSpec], now: datetime) -> str:
    listed = ", ".join(f"{spec.display_name} ({spec.symbol})" for spec in universe)
    return f"Current time: {_iso(now)}. Markets: {listed}."


def _iso(now: datetime) -> str:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone aware")
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sources(transcript: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for blob in _response_blobs(transcript):
        _walk_sources(blob, found)
    return found


def _response_blobs(transcript: dict[str, Any]) -> list[dict[str, object]]:
    responses = _objects(transcript.get("responses"))
    if responses:
        return _mappings(responses)
    response = transcript.get("response")
    if isinstance(response, dict):
        return [cast(dict[str, object], response)]
    return []


def _objects(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return list(cast(list[object], value))


def _mappings(values: list[object]) -> list[dict[str, object]]:
    return [cast(dict[str, object], value) for value in values if isinstance(value, dict)]


def _walk_sources(value: object, found: list[str]) -> None:
    if isinstance(value, dict):
        block = cast(dict[str, object], value)
        _collect_urls(block, found)
        for child in block.values():
            _walk_sources(child, found)
        return
    if isinstance(value, list):
        for item in cast(list[object], value):
            _walk_sources(item, found)


def _collect_urls(block: dict[str, object], found: list[str]) -> None:
    kind = block.get("type")
    if kind == "web_search_result":
        _remember(block.get("url"), found)
        return
    if kind != "text":
        return
    for citation in _mappings(_objects(block.get("citations"))):
        if citation.get("type") != "web_search_result_location":
            continue
        _remember(citation.get("url"), found)


def _remember(url: object, found: list[str]) -> None:
    if isinstance(url, str) and url not in found:
        found.append(url)
