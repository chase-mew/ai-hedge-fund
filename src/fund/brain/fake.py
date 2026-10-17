# pyright: strict
"""Offline Claude stand in. prime(snapshot) supplies the book; usage cost is zero."""

from __future__ import annotations

from typing import Any

from fund.brain.llm import LlmResult
from fund.config import ModelSpec, NewsSettings
from fund.contracts import AnalystReport, AnalystView, LlmUsage, MarketDecision, PortfolioDecision, Snapshot

_DIGEST = (
    "Macro\n"
    "Nothing notable.\n"
    "\n"
    "Equities\n"
    "Nothing notable.\n"
    "\n"
    "Commodities and FX\n"
    "Nothing notable.\n"
    "\n"
    "Crypto\n"
    "Nothing notable.\n"
)
_SOURCE = "https://example.com/digest"


class FakeClaude:
    """Drop in parse and search with a deterministic book and a fixed digest."""

    def __init__(self) -> None:
        self._snapshot: Snapshot | None = None

    def prime(self, snapshot: Snapshot) -> None:
        """Store the snapshot used for portfolio decisions and analyst views."""
        self._snapshot = snapshot

    async def parse[T](
        self,
        *,
        purpose: str,
        model: ModelSpec,
        system: str,
        user: str,
        output_type: type[T],
    ) -> LlmResult[T]:
        """Build a PortfolioDecision or a neutral AnalystReport without calling an API."""
        snapshot = self._require_snapshot()
        if output_type is PortfolioDecision:
            output: Any = _portfolio(snapshot)
        elif output_type is AnalystReport:
            output = _analyst(snapshot, user)
        else:
            raise TypeError(f"FakeClaude cannot build {output_type.__name__}.")
        return LlmResult(output, _zero_usage(purpose, model), _transcript(purpose, model, system, user), None)

    async def search(
        self,
        *,
        purpose: str,
        model: ModelSpec,
        system: str,
        user: str,
        news: NewsSettings,
    ) -> LlmResult[str]:
        """Return a fixed digest and zero cost usage."""
        response = {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": _DIGEST,
                    "citations": [{"type": "web_search_result_location", "url": _SOURCE}],
                }
            ],
        }
        transcript = {
            "request": {
                "purpose": purpose,
                "model": model.id,
                "system": system,
                "user": user,
                "max_searches": news.max_searches,
            },
            "response": response,
        }
        return LlmResult(_DIGEST, _zero_usage(purpose, model), transcript, None)

    def _require_snapshot(self) -> Snapshot:
        if self._snapshot is None:
            raise ValueError("Call FakeClaude.prime(snapshot) before requesting a decision.")
        return self._snapshot


def _portfolio(snapshot: Snapshot) -> PortfolioDecision:
    decisions = [
        MarketDecision(
            symbol=market.symbol,
            thesis=f"The 20 day average gap is {market.sma20_gap * 100:+.2f} percent.",
            invalidation="The 20 day average gap changes sign.",
            prob_up_24h=_probability(market.sma20_gap),
            target_weight=_target(_probability(market.sma20_gap)),
            stop_loss_pct=5.0,
        )
        for market in snapshot.markets
    ]
    return PortfolioDecision(
        market_view="Positions follow the 20 day average gap, scaled down and capped.",
        decisions=decisions,
    )


def _analyst(snapshot: Snapshot, user: str) -> AnalystReport:
    views = [
        AnalystView(
            symbol=market.symbol,
            key_points=["No edge versus the 20 day average."],
            risks=["The view can be wrong."],
            stance="neutral",
            prob_up_24h=0.5,
        )
        for market in snapshot.markets
        if market.symbol in user
    ]
    return AnalystReport(summary="No directional view for this group.", views=views)


def _probability(gap: float) -> float:
    return 0.5 + _clip(gap * 2, -0.15, 0.15)


def _target(probability: float) -> float:
    return _clip((probability - 0.5) * 0.8, -0.1, 0.1)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _zero_usage(purpose: str, model: ModelSpec) -> LlmUsage:
    return LlmUsage(
        purpose=purpose,
        model_id=model.id,
        input_tokens=0,
        output_tokens=0,
        cache_write_tokens=0,
        cache_read_tokens=0,
        web_searches=0,
        cost_usd=0.0,
    )


def _transcript(purpose: str, model: ModelSpec, system: str, user: str) -> dict[str, Any]:
    return {
        "request": {"purpose": purpose, "model": model.id, "system": system, "user": user},
        "response": {"stop_reason": "end_turn"},
    }
