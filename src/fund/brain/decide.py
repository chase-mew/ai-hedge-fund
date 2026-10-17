# pyright: strict
"""Single model and analyst team decision flows."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from fund.brain.llm import Llm, LlmResult
from fund.brain.prompts import SYSTEM_ANALYST, SYSTEM_PM, SYSTEM_TEAM_PM, render_limits
from fund.brain.render import render_analyst_reports, render_snapshot
from fund.config import AgentConfig, RiskPolicy
from fund.contracts import (
    AnalystReport,
    AssetClass,
    LlmUsage,
    MarketFeatures,
    PortfolioDecision,
    PortfolioState,
    Snapshot,
    TrackRecord,
)

logger = logging.getLogger(__name__)

_GROUPS: dict[str, frozenset[AssetClass]] = {
    "equities": frozenset({AssetClass.EQUITY_INDEX, AssetClass.STOCK}),
    "macro": frozenset({AssetClass.COMMODITY, AssetClass.FX}),
    "crypto": frozenset({AssetClass.CRYPTO}),
}


@dataclass(frozen=True)
class DecisionResult:
    """The portfolio decision, every call's usage, and any analyst or model error."""

    decision: PortfolioDecision | None
    usage: list[LlmUsage]
    transcripts: list[dict[str, Any]]
    error: str | None


async def decide(
    config: AgentConfig,
    claude: Llm,
    snapshot: Snapshot,
    portfolio: PortfolioState,
    track: TrackRecord,
    policy: RiskPolicy,
) -> DecisionResult:
    """Run one single parse, or three analyst parses and then the team portfolio manager."""
    if config.style == "team":
        result = await _team(config, claude, snapshot, portfolio, track, policy)
    else:
        result = await _single(config, claude, snapshot, portfolio, track, policy)
    state = "ok" if result.decision is not None else "failed"
    logger.info("Decision %s (%s) %s", config.config_id, config.style, state)
    return result


async def _single(
    config: AgentConfig,
    claude: Llm,
    snapshot: Snapshot,
    portfolio: PortfolioState,
    track: TrackRecord,
    policy: RiskPolicy,
) -> DecisionResult:
    result = await claude.parse(
        purpose=f"decision:{config.config_id}",
        model=config.model,
        system=SYSTEM_PM.format(limits=render_limits(policy)),
        user=render_snapshot(snapshot, portfolio, track),
        output_type=PortfolioDecision,
    )
    return DecisionResult(
        decision=result.output,
        usage=[result.usage],
        transcripts=[result.transcript],
        error=result.error,
    )


async def _team(
    config: AgentConfig,
    claude: Llm,
    snapshot: Snapshot,
    portfolio: PortfolioState,
    track: TrackRecord,
    policy: RiskPolicy,
) -> DecisionResult:
    groups = _group_markets(snapshot)
    analyst_calls = (
        _analyst(config, claude, snapshot, portfolio, track, name, markets) for name, markets in groups
    )
    results = await asyncio.gather(*analyst_calls)
    reports: dict[str, AnalystReport] = {}
    notes: list[str] = []
    usage: list[LlmUsage] = []
    transcripts: list[dict[str, Any]] = []
    for (name, _), result in zip(groups, results, strict=True):
        usage.append(result.usage)
        transcripts.append(result.transcript)
        if result.error is not None or result.output is None:
            notes.append(f"{name} analyst failed: {result.error or 'no report'}")
        else:
            reports[name] = result.output
    limits = render_limits(policy)
    user = render_snapshot(snapshot, portfolio, track).rstrip() + "\n\n" + render_analyst_reports(reports)
    pm = await claude.parse(
        purpose=f"decision:{config.config_id}",
        model=config.model,
        system=SYSTEM_TEAM_PM.format(limits=limits),
        user=user,
        output_type=PortfolioDecision,
    )
    usage.append(pm.usage)
    transcripts.append(pm.transcript)
    if pm.error is not None or pm.output is None:
        notes.append(pm.error or "The portfolio manager returned no decision.")
    error = "; ".join(notes) if notes else None
    return DecisionResult(decision=pm.output, usage=usage, transcripts=transcripts, error=error)


async def _analyst(
    config: AgentConfig,
    claude: Llm,
    snapshot: Snapshot,
    portfolio: PortfolioState,
    track: TrackRecord,
    name: str,
    markets: list[MarketFeatures],
) -> LlmResult[AnalystReport]:
    listed = ", ".join(f"{market.display_name} ({market.symbol})" for market in markets)
    symbols = [market.symbol for market in markets]
    return await claude.parse(
        purpose=f"analyst:{config.config_id}:{name}",
        model=config.model,
        system=SYSTEM_ANALYST.format(group=name, markets=listed),
        user=render_snapshot(snapshot, portfolio, track, markets=symbols),
        output_type=AnalystReport,
    )


def _group_markets(snapshot: Snapshot) -> list[tuple[str, list[MarketFeatures]]]:
    grouped: list[tuple[str, list[MarketFeatures]]] = []
    for name, classes in _GROUPS.items():
        markets = [market for market in snapshot.markets if market.asset_class in classes]
        if markets:
            grouped.append((name, markets))
    return grouped
