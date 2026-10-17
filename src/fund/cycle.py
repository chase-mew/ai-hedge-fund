"""One fund cycle. Data, decisions, the risk gate, paper books, and the optional live venue."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fund.baselines import BASELINES, as_gate_result, baseline_targets
from fund.brain.decide import DecisionResult, decide
from fund.brain.llm import Llm
from fund.brain.news import compile_news
from fund.clock import cycle_id, cycle_slot, session_state, utc_day
from fund.config import AgentConfig, FundConfig, Secrets
from fund.contracts import (
    AccountSnapshot,
    Candle,
    ConfigRun,
    CycleRecord,
    EquityPoint,
    Forecast,
    GateResult,
    LlmUsage,
    MarketMeta,
    MarketQuote,
    Mode,
    OrderOutcome,
    PaperAccount,
    PortfolioState,
    Position,
    Snapshot,
)
from fund.data.base import MarketData
from fund.data.snapshot import build_snapshot
from fund.execution.executor import execute, plan_orders
from fund.journal.store import Journal, record_flips, resolve, update_bookkeeping
from fund.risk.gate import apply_gate
from fund.venues.base import Venue
from fund.venues.paper import PaperVenue, accrue_funding, new_paper_account, paper_equity, process_stops

logger = logging.getLogger(__name__)


@dataclass
class CycleContext:
    """Everything one cycle needs. now is injected so tests do not read the clock."""

    config: FundConfig
    secrets: Secrets
    journal: Journal
    data: MarketData
    claude: Llm
    mode: Mode
    now: datetime
    live_venue: Venue | None = None
    approve: Any = None
    force: bool = False


def run_cycle(ctx: CycleContext) -> CycleRecord:
    """Run one cycle, store it, and return the record. Market and live failures stay in errors."""
    now = _aware(ctx.now)
    slot = cycle_id(cycle_slot(now, ctx.config.fund.cycle_hours))
    if ctx.journal.has_cycle(slot) and not ctx.force:
        logger.info("Cycle %s is already stored.", slot)
        stored = _stored(ctx.journal, slot)
        if stored is not None:
            return stored
    if ctx.journal.halt_state() is not None:
        record = _halted(ctx, slot, now)
        ctx.journal.write_cycle(record)
        logger.info("Cycle %s recorded as halted.", slot)
        return record
    try:
        snapshot = build_snapshot(ctx.data, ctx.config.universe, now, ctx.config.fund.cycle_hours)
    except Exception as exc:
        logger.exception("Snapshot failed.")
        record = _failed(ctx, slot, now, f"Market data failed: {exc}")
        ctx.journal.write_cycle(record)
        return record
    try:
        return _complete(ctx, snapshot, now)
    except Exception as exc:
        logger.exception("Cycle failed after the snapshot.")
        record = _failed(ctx, snapshot.cycle_id, now, f"Cycle failed: {exc}", snapshot)
        ctx.journal.write_cycle(record)
        return record


def cycle_failed(record: CycleRecord) -> bool:
    """Return whether the CLI should exit non zero. A halt by itself is a stored outcome."""
    return any(item != "halted" for item in record.errors)


def _complete(ctx: CycleContext, snapshot: Snapshot, now: datetime) -> CycleRecord:
    marks = {market.symbol: market.mark for market in snapshot.markets}
    symbols = [spec.symbol for spec in ctx.config.universe]
    meta = ctx.data.meta(symbols)
    accounts = _housekeep(ctx, symbols, marks, now)
    thought = asyncio.run(_think(ctx, snapshot, accounts, marks))
    snapshot, news_usage, news_transcript, selected, results = thought
    errors: list[str] = []
    runs = _paper_runs(ctx, snapshot, accounts, marks, meta, selected, results, news_usage, now)
    _live(ctx, snapshot, marks, runs, errors, now)
    _baselines(ctx, snapshot, accounts, marks, meta, errors, now)
    _resolve_due(ctx, now)
    forecasts = _forecasts(snapshot, runs)
    if forecasts:
        ctx.journal.append_forecasts(forecasts)
    points = [_paper_point(account, marks, snapshot.cycle_id, now) for account in accounts.values()]
    ctx.journal.append_equity(points)
    if news_transcript is not None:
        ctx.journal.write_transcripts(snapshot.cycle_id, "news", [news_transcript])
    record = CycleRecord(
        cycle_id=snapshot.cycle_id,
        mode=ctx.mode,
        started_at=now,
        finished_at=now,
        snapshot=snapshot,
        news_usage=news_usage,
        runs=runs,
        errors=errors,
    )
    ctx.journal.write_cycle(record)
    logger.info("Cycle %s stored with %s config runs.", record.cycle_id, len(runs))
    return record


async def _think(
    ctx: CycleContext,
    snapshot: Snapshot,
    accounts: dict[str, PaperAccount],
    marks: dict[str, float],
) -> tuple[
    Snapshot,
    LlmUsage | None,
    dict[str, Any] | None,
    list[AgentConfig],
    list[DecisionResult | Exception],
]:
    day = utc_day(ctx.now)
    spent = ctx.journal.llm_spend_on(day)
    budget = ctx.config.fund.daily_llm_budget_usd
    news_usage: LlmUsage | None = None
    news_transcript: dict[str, Any] | None = None
    if ctx.config.news.enabled and spent < budget:
        digest, news_usage, news_transcript = await compile_news(
            ctx.claude,
            ctx.config.catalog[ctx.config.news.model],
            ctx.config.news,
            ctx.config.universe,
            ctx.now,
        )
        if digest is not None:
            snapshot = snapshot.model_copy(update={"news": digest})
        if news_usage is not None:
            spent += news_usage.cost_usd
    selected = list(ctx.config.configs)
    if spent > budget:
        logger.info("Daily model budget is spent. Only the champion will run.")
        selected = [ctx.config.champion]
    prime = getattr(ctx.claude, "prime", None)
    if callable(prime):
        prime(snapshot)
    tasks = [_one_decision(ctx, snapshot, accounts, marks, config) for config in selected]
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[DecisionResult | Exception] = []
    for item in raw:
        if isinstance(item, Exception):
            results.append(item)
        elif isinstance(item, BaseException):
            raise item
        else:
            results.append(item)
    return snapshot, news_usage, news_transcript, selected, results


async def _one_decision(
    ctx: CycleContext,
    snapshot: Snapshot,
    accounts: dict[str, PaperAccount],
    marks: dict[str, float],
    config: AgentConfig,
) -> DecisionResult:
    portfolio = _state(accounts[_portfolio_id(config.config_id)], marks, ctx.now)
    track = ctx.journal.track_record(config.config_id, snapshot)
    return await decide(config, ctx.claude, snapshot, portfolio, track, ctx.config.risk)


def _paper_runs(
    ctx: CycleContext,
    snapshot: Snapshot,
    accounts: dict[str, PaperAccount],
    marks: dict[str, float],
    meta: dict[str, MarketMeta],
    selected: list[AgentConfig],
    results: list[DecisionResult | Exception],
    news_usage: LlmUsage | None,
    now: datetime,
) -> list[ConfigRun]:
    runs: list[ConfigRun] = []
    for config, result in zip(selected, results, strict=True):
        portfolio_id = _portfolio_id(config.config_id)
        champion = config.config_id == ctx.config.champion.config_id
        extra = news_usage.cost_usd if news_usage is not None and champion else 0.0
        try:
            if isinstance(result, Exception):
                raise result
            runs.append(_trade_config(ctx, snapshot, accounts, marks, meta, config, result, extra, now))
        except Exception as exc:
            logger.exception("Config %s failed.", config.config_id)
            runs.append(_error_run(config.config_id, portfolio_id, accounts[portfolio_id], marks, str(exc)))
    return runs


def _trade_config(
    ctx: CycleContext,
    snapshot: Snapshot,
    accounts: dict[str, PaperAccount],
    marks: dict[str, float],
    meta: dict[str, MarketMeta],
    config: AgentConfig,
    result: DecisionResult,
    extra_cost: float,
    now: datetime,
) -> ConfigRun:
    portfolio_id = _portfolio_id(config.config_id)
    account = accounts[portfolio_id]
    before = paper_equity(account, marks)
    state = _state(account, marks, now)
    gate = apply_gate(result.decision, state, snapshot, ctx.config.universe, ctx.config.risk, now)
    booked = account.model_copy(update={"bookkeeping": state.bookkeeping})
    venue = PaperVenue(booked, marks, meta, ctx.config.paper, now)
    outcomes = execute(gate, venue, ctx.config.execution, marks)
    cost = extra_cost + sum(item.cost_usd for item in result.usage)
    saved = _stamp(venue.state(), marks, cost, now)
    accounts[portfolio_id] = saved
    ctx.journal.save_paper(saved)
    path = None
    if result.transcripts:
        path = ctx.journal.write_transcripts(snapshot.cycle_id, config.config_id, result.transcripts)
    return ConfigRun(
        config_id=config.config_id,
        portfolio_id=portfolio_id,
        decision=result.decision,
        error=result.error,
        gate=gate,
        outcomes=outcomes,
        usage=list(result.usage),
        transcript_path=path,
        equity_before=before,
        equity_after=paper_equity(saved, marks),
    )


def _live(
    ctx: CycleContext,
    snapshot: Snapshot,
    marks: dict[str, float],
    runs: list[ConfigRun],
    errors: list[str],
    now: datetime,
) -> None:
    if ctx.mode == "paper" or ctx.config.fund.autonomy == "paper":
        return
    if ctx.live_venue is None:
        errors.append("No live venue is configured, so no live orders were sent.")
        return
    champion = _champion_run(runs, ctx.config.champion.config_id)
    if champion is None or champion.decision is None:
        errors.append("The champion has no decision, so no live orders were sent.")
        return
    venue = ctx.live_venue
    try:
        before = venue.account()
    except Exception as exc:
        logger.exception("Live account read failed.")
        errors.append(f"Live account read failed: {exc}")
        return
    ceiling = ctx.config.execution.max_live_equity_usd
    if before.equity > ceiling:
        errors.append(f"Live equity {before.equity:.2f} exceeds the ceiling of {ceiling:.2f}.")
        return
    book = update_bookkeeping(ctx.journal.load_live_bookkeeping(), before.equity, now)
    state = PortfolioState(portfolio_id="live", as_of=now, account=before, bookkeeping=book)
    gate = apply_gate(champion.decision, state, snapshot, ctx.config.universe, ctx.config.risk, now)
    if ctx.config.fund.autonomy == "approve" and ctx.approve is None:
        errors.append("Approve mode needs a person at a terminal. No live orders were sent.")
        ctx.journal.save_live_bookkeeping(book)
        return
    if ctx.config.fund.autonomy == "approve" and not _approved(ctx, gate, venue):
        logger.info("Live orders were declined.")
        ctx.journal.save_live_bookkeeping(book)
        return
    try:
        outcomes = execute(gate, venue, ctx.config.execution, marks)
        outcomes = _merge_fees(outcomes, _fees(venue, now))
    except Exception as exc:
        logger.exception("Live execution failed.")
        errors.append(f"Live execution failed: {exc}")
        ctx.journal.save_live_bookkeeping(book)
        return
    after = venue.account()
    book = record_flips(update_bookkeeping(book, after.equity, now), before, after, now)
    ctx.journal.save_live_bookkeeping(book)
    _remember_live(ctx, snapshot.cycle_id, after, outcomes, now)
    if gate.halt:
        ctx.journal.set_halt(gate.halt_reason or "drawdown", now)
        _flatten(venue, errors)


def _baselines(
    ctx: CycleContext,
    snapshot: Snapshot,
    accounts: dict[str, PaperAccount],
    marks: dict[str, float],
    meta: dict[str, MarketMeta],
    errors: list[str],
    now: datetime,
) -> None:
    for name in BASELINES:
        portfolio_id = f"baseline:{name}"
        try:
            account = accounts[portfolio_id]
            state = _state(account, marks, now)
            gate = as_gate_result(baseline_targets(name, snapshot, now), state)
            venue = PaperVenue(
                account.model_copy(update={"bookkeeping": state.bookkeeping}),
                marks,
                meta,
                ctx.config.paper,
                now,
            )
            execute(gate, venue, ctx.config.execution, marks)
            saved = _stamp(venue.state(), marks, 0.0, now)
            accounts[portfolio_id] = saved
            ctx.journal.save_paper(saved)
        except Exception as exc:
            logger.exception("Baseline %s failed.", name)
            errors.append(f"Baseline {name} failed: {exc}")


def _housekeep(
    ctx: CycleContext, symbols: list[str], marks: dict[str, float], now: datetime
) -> dict[str, PaperAccount]:
    ids = [_portfolio_id(item.config_id) for item in ctx.config.configs]
    ids.extend(f"baseline:{name}" for name in BASELINES)
    loaded = {portfolio_id: ctx.journal.load_paper(portfolio_id) for portfolio_id in ids}
    present = [account for account in loaded.values() if account is not None]
    oldest = min((account.updated_at for account in present), default=now)
    quotes = ctx.data.quotes(symbols)
    hourly = _hourly(ctx.data, symbols, oldest, now)
    accounts: dict[str, PaperAccount] = {}
    for portfolio_id in ids:
        account = loaded[portfolio_id]
        if account is None:
            account = new_paper_account(portfolio_id, ctx.config.fund.paper_starting_equity, now)
        else:
            account = _accrue_then_stops(account, quotes, hourly, ctx, now)
        equity = paper_equity(account, marks)
        book = update_bookkeeping(account.bookkeeping, equity, now)
        account = account.model_copy(
            update={"bookkeeping": book, "last_marks": {**account.last_marks, **marks}}
        )
        accounts[portfolio_id] = account
        ctx.journal.save_paper(account)
    return accounts


def _accrue_then_stops(
    account: PaperAccount,
    quotes: dict[str, MarketQuote],
    hourly: dict[str, list[Candle]],
    ctx: CycleContext,
    now: datetime,
) -> PaperAccount:
    """Charge funding across the open interval, then fill stops. Funding uses the pre stop clock."""
    funded = accrue_funding(account, quotes, now)
    return process_stops(funded, hourly, ctx.config.paper, now)


def _hourly(data: MarketData, symbols: list[str], start: datetime, end: datetime) -> dict[str, list[Candle]]:
    found: dict[str, list[Candle]] = {}
    for symbol in symbols:
        found[symbol] = data.candles(symbol, "1h", start, end)
    return found


def _resolve_due(ctx: CycleContext, now: datetime) -> None:
    due = ctx.journal.unresolved_forecasts(now)
    if not due:
        return
    start = min(item.resolves_at for item in due) - timedelta(hours=1)
    symbols = sorted({item.symbol for item in due})
    hourly = _hourly(ctx.data, symbols, start, now)
    found = [row for item in due if (row := resolve(item, hourly.get(item.symbol, []), now)) is not None]
    if found:
        ctx.journal.append_resolutions(found)


def _forecasts(snapshot: Snapshot, runs: list[ConfigRun]) -> list[Forecast]:
    marks = {market.symbol: market.mark for market in snapshot.markets}
    resolves_at = snapshot.as_of + timedelta(hours=24)
    rows: list[Forecast] = []
    for run in runs:
        if run.decision is None:
            continue
        for item in run.decision.decisions:
            mark = marks.get(item.symbol)
            if mark is None:
                continue
            rows.append(
                Forecast(
                    forecast_id=f"{snapshot.cycle_id}|{run.config_id}|{item.symbol}",
                    cycle_id=snapshot.cycle_id,
                    config_id=run.config_id,
                    symbol=item.symbol,
                    made_at=snapshot.as_of,
                    resolves_at=resolves_at,
                    prob_up=item.prob_up_24h,
                    ref_price=mark,
                )
            )
    return rows


def _state(account: PaperAccount, marks: dict[str, float], now: datetime) -> PortfolioState:
    snap = _marked(account, marks)
    book = update_bookkeeping(account.bookkeeping, snap.equity, now)
    return PortfolioState(portfolio_id=account.portfolio_id, as_of=now, account=snap, bookkeeping=book)


def _marked(account: PaperAccount, marks: dict[str, float]) -> AccountSnapshot:
    positions = {
        symbol: Position(
            symbol=symbol,
            size=position.size,
            entry_px=position.entry_px,
            mark=marks.get(symbol, account.last_marks.get(symbol, position.entry_px)),
        )
        for symbol, position in account.positions.items()
    }
    return AccountSnapshot(equity=paper_equity(account, marks), positions=positions)


def _stamp(account: PaperAccount, marks: dict[str, float], cost: float, now: datetime) -> PaperAccount:
    equity = paper_equity(account, marks)
    book = update_bookkeeping(account.bookkeeping, equity, now)
    return account.model_copy(
        update={
            "bookkeeping": book,
            "last_marks": {**account.last_marks, **marks},
            "llm_cost_cum": account.llm_cost_cum + cost,
        }
    )


def _paper_point(
    account: PaperAccount, marks: dict[str, float], cycle_id_value: str, now: datetime
) -> EquityPoint:
    equity = paper_equity(account, marks)
    gross, net = _exposure(_marked(account, marks))
    return EquityPoint(
        at=now,
        cycle_id=cycle_id_value,
        portfolio_id=account.portfolio_id,
        equity=equity,
        gross=gross,
        net=net,
        fees_cum=account.fees_cum,
        funding_cum=account.funding_cum,
        llm_cost_cum=account.llm_cost_cum,
    )


def _remember_live(
    ctx: CycleContext,
    cycle_id_value: str,
    account: AccountSnapshot,
    outcomes: list[OrderOutcome],
    now: datetime,
) -> None:
    prior = [point for point in ctx.journal.load_equity() if point.portfolio_id == "live"]
    fees = (prior[-1].fees_cum if prior else 0.0) + sum(item.fee_usd for item in outcomes)
    gross, net = _exposure(account)
    ctx.journal.append_equity(
        [
            EquityPoint(
                at=now,
                cycle_id=cycle_id_value,
                portfolio_id="live",
                equity=account.equity,
                gross=gross,
                net=net,
                fees_cum=fees,
                funding_cum=prior[-1].funding_cum if prior else 0.0,
                llm_cost_cum=0.0,
            )
        ]
    )


def _exposure(account: AccountSnapshot) -> tuple[float, float]:
    if account.equity <= 0:
        return 0.0, 0.0
    weights = [position.notional / account.equity for position in account.positions.values()]
    return sum(abs(weight) for weight in weights), sum(weights)


def _approved(ctx: CycleContext, gate: GateResult, venue: Venue) -> bool:
    plans = plan_orders(
        gate,
        venue.account(),
        venue.marks(),
        venue.market_meta(),
        ctx.config.execution.min_order_usd,
    )
    approve = ctx.approve
    if not callable(approve):
        return False
    return bool(approve(gate, plans))


def _fees(venue: Venue, now: datetime) -> dict[str, float]:
    reader = getattr(venue, "fees_since", None)
    if not callable(reader):
        return {}
    found = reader(now)
    if not isinstance(found, dict):
        return {}
    return {str(key): float(value) for key, value in found.items()}


def _merge_fees(outcomes: list[OrderOutcome], fees: dict[str, float]) -> list[OrderOutcome]:
    merged: list[OrderOutcome] = []
    for item in outcomes:
        extra = 0.0 if item.order_id is None else fees.get(item.order_id, 0.0)
        if extra:
            merged.append(item.model_copy(update={"fee_usd": item.fee_usd + extra}))
        else:
            merged.append(item)
    return merged


def _flatten(venue: Venue, errors: list[str]) -> None:
    try:
        account = venue.account()
        for symbol, position in account.positions.items():
            if position.size != 0:
                venue.close_position(symbol)
            venue.sync_stop(symbol, None)
    except Exception as exc:
        logger.exception("Flatten failed.")
        errors.append(f"Flatten failed: {exc}")


def _champion_run(runs: list[ConfigRun], config_id: str) -> ConfigRun | None:
    for run in runs:
        if run.config_id == config_id:
            return run
    return None


def _error_run(
    config_id: str, portfolio_id: str, account: PaperAccount, marks: dict[str, float], message: str
) -> ConfigRun:
    equity = paper_equity(account, marks)
    return ConfigRun(
        config_id=config_id,
        portfolio_id=portfolio_id,
        decision=None,
        error=message,
        gate=None,
        outcomes=[],
        usage=[],
        transcript_path=None,
        equity_before=equity,
        equity_after=equity,
    )


def _portfolio_id(config_id: str) -> str:
    return f"config:{config_id}"


def _halted(ctx: CycleContext, slot: str, now: datetime) -> CycleRecord:
    return _failed(ctx, slot, now, "halted")


def _failed(
    ctx: CycleContext, slot: str, now: datetime, message: str, snapshot: Snapshot | None = None
) -> CycleRecord:
    if snapshot is None:
        snapshot = Snapshot(cycle_id=slot, as_of=now, session=session_state(now), markets=[])
    return CycleRecord(
        cycle_id=slot,
        mode=ctx.mode,
        started_at=now,
        finished_at=now,
        snapshot=snapshot,
        news_usage=None,
        runs=[],
        errors=[message],
    )


def _stored(journal: Journal, slot: str) -> CycleRecord | None:
    for record in journal.load_cycles():
        if record.cycle_id == slot:
            return record
    return None


def _aware(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone aware")
    return now.astimezone(UTC)
