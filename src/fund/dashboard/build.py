# pyright: strict
"""Render the static research dashboard."""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from jinja2 import Environment, FileSystemLoader

from fund.config import FundConfig
from fund.contracts import (
    AssetClass,
    Bookkeeping,
    ConfigRun,
    CycleRecord,
    EquityPoint,
    ForecastScore,
    GatedTarget,
    GateResult,
    LlmUsage,
    MarketDecision,
    MarketSpec,
    OrderOutcome,
    PaperAccount,
    PaperPosition,
    PortfolioDecision,
    PortfolioScore,
    ScoreReport,
    Snapshot,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
STYLESHEET = ROOT / "static" / "style.css"
TAGLINE = "Claude proposes, code disposes. Every decision is journaled, scored and published."
SP500_PORTFOLIO = "baseline:sp500"
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_CLASS_LABELS: dict[AssetClass, str] = {
    AssetClass.EQUITY_INDEX: "Equity index",
    AssetClass.STOCK: "Stock",
    AssetClass.COMMODITY: "Commodity",
    AssetClass.FX: "FX",
    AssetClass.CRYPTO: "Crypto",
}


class JournalView(Protocol):
    def load_cycles(self, limit: int | None = None) -> list[CycleRecord]: ...

    def load_equity(self) -> list[EquityPoint]: ...

    def load_scores(self) -> ScoreReport | None: ...

    def halt_state(self) -> dict[str, str] | None: ...

    def load_paper(self, portfolio_id: str) -> PaperAccount | None: ...

    def load_live_bookkeeping(self) -> Bookkeeping | None: ...


@dataclass(frozen=True)
class Figure:
    label: str
    value: str
    css_class: str


@dataclass(frozen=True)
class LeaderRow:
    portfolio: str
    kind: str
    total_return: str
    net_return: str
    drawdown: str
    sharpe: str
    fees: str
    funding: str
    cycles: str
    return_class: str
    net_class: str
    drawdown_class: str


@dataclass(frozen=True)
class BrierRow:
    config_id: str
    resolved: str
    brier: str
    skill: str
    skill_class: str


@dataclass(frozen=True)
class PositionRow:
    market: str
    asset_class: str
    weight: str
    entry: str
    unrealised: str
    stop: str
    thesis: str
    weight_class: str
    unrealised_class: str


@dataclass(frozen=True)
class DecisionRow:
    market: str
    requested: str
    approved: str
    status: str
    reasons: str
    probability: str
    thesis: str
    requested_class: str
    approved_class: str


@dataclass(frozen=True)
class CycleLink:
    href: str
    when: str
    mode: str
    summary: str
    orders: str
    cost: str


@dataclass(frozen=True)
class IndexPage:
    title: str
    asset_prefix: str
    name: str
    tagline: str
    mode: str | None
    last_cycle: str | None
    halt: str | None
    empty: bool
    built: str
    headline: list[Figure]
    has_chart: bool
    chart: dict[str, object]
    leaders: list[LeaderRow]
    has_calibration: bool
    calibration: dict[str, object]
    calibration_note: str
    brier: list[BrierRow]
    account_label: str
    positions_note: str | None
    positions: list[PositionRow]
    decision_note: str | None
    market_view: str
    decisions: list[DecisionRow]
    reasoning: str
    cycles: list[CycleLink]


@dataclass(frozen=True)
class SourceLink:
    href: str | None
    label: str


@dataclass(frozen=True)
class SnapshotRow:
    market: str
    asset_class: str
    status: str
    mark: str
    ret_24h: str
    ret_7d: str
    ret_30d: str
    vol: str
    funding: str
    open_interest: str
    volume: str
    ret_24h_class: str
    ret_7d_class: str
    ret_30d_class: str
    funding_class: str


@dataclass(frozen=True)
class OrderRow:
    market: str
    side: str
    requested: str
    filled: str
    price: str
    fee: str
    status: str


@dataclass(frozen=True)
class CostRow:
    purpose: str
    model: str
    input_tokens: str
    output_tokens: str
    cache_write: str
    cache_read: str
    searches: str
    cost: str


@dataclass(frozen=True)
class ConfigBlock:
    config_id: str
    error: str | None
    market_view: str | None
    decisions: list[DecisionRow]
    orders: list[OrderRow]
    reasoning: str
    cost: str


@dataclass(frozen=True)
class CyclePage:
    title: str
    asset_prefix: str
    name: str
    built: str
    when: str
    mode: str
    cycle_id: str
    session_line: str
    markets: list[SnapshotRow]
    news_text: str | None
    news_compiled: str | None
    sources: list[SourceLink]
    configs: list[ConfigBlock]
    costs: list[CostRow]
    cost_total: str
    errors: list[str]


def build_site(journal: JournalView, config: FundConfig, out_dir: Path, now: datetime) -> None:
    """Render the static dashboard into out_dir."""
    moment = _aware(now)
    out_dir.mkdir(parents=True, exist_ok=True)
    cycles = sorted(journal.load_cycles(), key=lambda item: item.started_at)
    equity = sorted(journal.load_equity(), key=lambda item: (item.at, item.portfolio_id))
    scores = journal.load_scores()
    env = _environment()
    page = _index_page(journal, config, cycles, equity, scores, moment)
    (out_dir / "index.html").write_text(env.get_template("index.html.j2").render(page=page), encoding="utf-8")
    if cycles:
        cycle_dir = out_dir / "cycles"
        cycle_dir.mkdir(exist_ok=True)
        template = env.get_template("cycle.html.j2")
        for record in cycles[-30:]:
            rendered = template.render(page=_cycle_page(config, record, moment))
            (cycle_dir / f"{_slug(record.cycle_id)}.html").write_text(rendered, encoding="utf-8")
    _write_data(out_dir, moment, scores, equity)
    shutil.copyfile(STYLESHEET, out_dir / "style.css")


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _index_page(
    journal: JournalView,
    config: FundConfig,
    cycles: list[CycleRecord],
    equity: list[EquityPoint],
    scores: ScoreReport | None,
    moment: datetime,
) -> IndexPage:
    name = config.fund.name
    halt = _halt_text(journal.halt_state())
    built = _when(moment)
    if not cycles:
        return _blank_page(name, halt, built)
    latest = cycles[-1]
    config_id, portfolio_id = _champion_ids(config)
    grouped = _grouped(equity)
    account, account_label, paper = _select_account(journal, portfolio_id)
    run = _run(latest, config_id)
    decision_note, market_view, decisions = _latest_decision(run, config.universe)
    has_calibration, calibration, calibration_note = _calibration(scores, config_id)
    return IndexPage(
        title=name,
        asset_prefix="",
        name=name,
        tagline=TAGLINE,
        mode=latest.mode.upper(),
        last_cycle=_when(latest.finished_at),
        halt=halt,
        empty=False,
        built=built,
        headline=_headline(config, scores, grouped, cycles, paper),
        has_chart=bool(grouped),
        chart=_chart(grouped, portfolio_id) if grouped else {},
        leaders=_leaders(scores),
        has_calibration=has_calibration,
        calibration=calibration,
        calibration_note=calibration_note,
        brier=_brier_rows(scores),
        account_label=account_label,
        positions_note=_positions_note(account),
        positions=_position_rows(account, config.universe, _theses(run)) if account is not None else [],
        decision_note=decision_note,
        market_view=market_view,
        decisions=decisions,
        reasoning=_read_transcript(run.transcript_path) if run is not None else "",
        cycles=_recent(cycles, config_id),
    )


def _blank_page(name: str, halt: str | None, built: str) -> IndexPage:
    return IndexPage(
        title=name,
        asset_prefix="",
        name=name,
        tagline=TAGLINE,
        mode=None,
        last_cycle=None,
        halt=halt,
        empty=True,
        built=built,
        headline=[],
        has_chart=False,
        chart={},
        leaders=[],
        has_calibration=False,
        calibration={},
        calibration_note="",
        brier=[],
        account_label="",
        positions_note=None,
        positions=[],
        decision_note=None,
        market_view="",
        decisions=[],
        reasoning="",
        cycles=[],
    )


def _cycle_page(config: FundConfig, record: CycleRecord, moment: datetime) -> CyclePage:
    news = record.snapshot.news
    costs, total = _cost_rows(record)
    return CyclePage(
        title=f"Cycle {_when(record.finished_at)}, {config.fund.name}",
        asset_prefix="../",
        name=config.fund.name,
        built=_when(moment),
        when=_when(record.finished_at),
        mode=record.mode.upper(),
        cycle_id=record.cycle_id,
        session_line=_session_line(record.snapshot),
        markets=_snapshot_rows(record.snapshot, config.universe),
        news_text=None if news is None else news.text,
        news_compiled=None if news is None else _when(news.compiled_at),
        sources=[] if news is None else [_source(item) for item in news.sources],
        configs=[_config_block(run, config.universe) for run in record.runs],
        costs=costs,
        cost_total=total,
        errors=list(record.errors),
    )


def _headline(
    config: FundConfig,
    scores: ScoreReport | None,
    grouped: dict[str, list[EquityPoint]],
    cycles: list[CycleRecord],
    paper: PaperAccount | None,
) -> list[Figure]:
    _, portfolio_id = _champion_ids(config)
    champion = _portfolio_score(scores, portfolio_id)
    points = grouped.get(portfolio_id, [])
    if champion is not None:
        equity: float | None = champion.equity
        ret: float | None = champion.total_return
    elif points:
        equity = points[-1].equity
        ret = _series_return(points)
    elif paper is not None:
        equity = _account_equity(paper)
        ret = None
    else:
        equity = None
        ret = None
    baseline = _portfolio_score(scores, SP500_PORTFOLIO)
    baseline_return = None if baseline is None else baseline.total_return
    forecast = _forecast_score(scores, config.champion.config_id)
    skill = None if forecast is None else forecast.brier_skill
    cost = _llm_total(cycles)
    if scores is not None:
        cost = sum(item.llm_cost for item in scores.portfolios)
    equity_text = _usd(equity) if equity is not None else "n/a"
    return [
        Figure("Champion equity", equity_text, ""),
        Figure("Champion return", _pct(ret), _tone(ret)),
        Figure("S&P 500 return", _pct(baseline_return), _tone(baseline_return)),
        Figure("Brier skill", _signed(skill, 3), _tone(skill)),
        Figure("Model cost", _usd(cost), ""),
    ]


def _leaders(scores: ScoreReport | None) -> list[LeaderRow]:
    if scores is None:
        return []
    ordered = sorted(scores.portfolios, key=lambda item: (-item.return_net_of_llm, item.portfolio_id))
    return [_leader(item) for item in ordered]


def _leader(item: PortfolioScore) -> LeaderRow:
    sharpe = f"{item.sharpe:.2f}" if item.sharpe is not None else "n/a, needs 30 cycles"
    return LeaderRow(
        portfolio=item.portfolio_id,
        kind=item.kind,
        total_return=_pct(item.total_return),
        net_return=_pct(item.return_net_of_llm),
        drawdown=_plain_pct(item.max_drawdown),
        sharpe=sharpe,
        fees=_usd(item.fees),
        funding=_usd(item.funding),
        cycles=str(item.cycles),
        return_class=_tone(item.total_return),
        net_class=_tone(item.return_net_of_llm),
        drawdown_class="negative" if item.max_drawdown > 0 else "",
    )


def _brier_rows(scores: ScoreReport | None) -> list[BrierRow]:
    if scores is None:
        return []
    return [
        BrierRow(
            config_id=item.config_id,
            resolved=str(item.resolved),
            brier=_digits(item.brier, 3),
            skill=_signed(item.brier_skill, 3),
            skill_class=_tone(item.brier_skill),
        )
        for item in scores.forecasts
    ]


def _calibration(scores: ScoreReport | None, config_id: str) -> tuple[bool, dict[str, object], str]:
    if scores is None:
        return False, {}, "No forecast scores for the champion yet."
    forecast = _forecast_score(scores, config_id)
    if forecast is None:
        return False, {}, "No forecast scores for the champion yet."
    if not forecast.calibration:
        return False, {}, "No calibration bins for the champion yet."
    points: list[dict[str, float]] = [
        {"x": item.mean_prob, "y": item.hit_rate} for item in forecast.calibration
    ]
    return True, {"points": points}, ""


def _chart(grouped: dict[str, list[EquityPoint]], champion_portfolio_id: str) -> dict[str, object]:
    times = sorted({point.at for series in grouped.values() for point in series})
    datasets: list[dict[str, object]] = []
    order = sorted(
        grouped, key=lambda portfolio_id: (_role_rank(portfolio_id, champion_portfolio_id), portfolio_id)
    )
    for portfolio_id in order:
        datasets.append(
            {
                "label": _series_label(portfolio_id, champion_portfolio_id),
                "role": _role(portfolio_id, champion_portfolio_id),
                "data": _indexed(grouped[portfolio_id], times),
            }
        )
    return {"labels": [_when(moment) for moment in times], "datasets": datasets}


def _select_account(
    journal: JournalView, champion_portfolio_id: str
) -> tuple[PaperAccount | None, str, PaperAccount | None]:
    book = journal.load_live_bookkeeping()
    live = journal.load_paper("live") if book is not None else None
    paper = journal.load_paper(champion_portfolio_id)
    if live is not None:
        return live, "Live account", paper
    if paper is not None:
        return paper, "Champion paper account", paper
    return None, "Champion paper account", None


def _positions_note(account: PaperAccount | None) -> str | None:
    if account is None:
        return "No account snapshot is available."
    if any(position.size != 0 for position in account.positions.values()):
        return None
    return "No open positions."


def _position_rows(
    account: PaperAccount, universe: list[MarketSpec], theses: dict[str, str]
) -> list[PositionRow]:
    specs = {item.symbol: item for item in universe}
    ordered = [item.symbol for item in universe if item.symbol in account.positions]
    ordered.extend(symbol for symbol in account.positions if symbol not in specs)
    equity = _account_equity(account)
    rows: list[PositionRow] = []
    for symbol in ordered:
        position = account.positions[symbol]
        if position.size == 0:
            continue
        spec = specs.get(symbol)
        mark = account.last_marks.get(symbol, position.entry_px)
        weight = 0.0 if equity == 0 else position.size * mark / equity
        change = _unrealised(position, mark)
        rows.append(
            PositionRow(
                market=spec.display_name if spec is not None else symbol,
                asset_class=_CLASS_LABELS[spec.asset_class] if spec is not None else "Unknown",
                weight=_weight(weight),
                entry=_price(position.entry_px),
                unrealised=_pct(change),
                stop=_price(position.stop.trigger_px) if position.stop is not None else "none",
                thesis=theses.get(symbol, ""),
                weight_class=_tone(weight),
                unrealised_class=_tone(change),
            )
        )
    return rows


def _latest_decision(
    run: ConfigRun | None, universe: list[MarketSpec]
) -> tuple[str | None, str, list[DecisionRow]]:
    if run is None or run.decision is None:
        return "No champion decision was recorded for the latest cycle.", "", []
    return None, run.decision.market_view, _decision_rows(run.decision, run.gate, universe)


def _recent(cycles: list[CycleRecord], config_id: str) -> list[CycleLink]:
    links: list[CycleLink] = []
    for record in reversed(cycles[-30:]):
        run = _run(record, config_id)
        links.append(
            CycleLink(
                href=f"cycles/{_slug(record.cycle_id)}.html",
                when=_when(record.finished_at),
                mode=record.mode.upper(),
                summary=_summary(run),
                orders=str(_order_count(run)),
                cost=_usd(_cycle_cost(record)),
            )
        )
    return links


def _config_block(run: ConfigRun, universe: list[MarketSpec]) -> ConfigBlock:
    market_view = None
    decisions: list[DecisionRow] = []
    if run.decision is not None or run.gate is not None:
        decisions = _decision_rows(run.decision, run.gate, universe)
        if run.decision is not None:
            market_view = run.decision.market_view
    return ConfigBlock(
        config_id=run.config_id,
        error=run.error,
        market_view=market_view,
        decisions=decisions,
        orders=[_order_row(item) for item in run.outcomes],
        reasoning=_read_transcript(run.transcript_path),
        cost=_usd(sum(item.cost_usd for item in run.usage)),
    )


def _decision_rows(
    decision: PortfolioDecision | None, gate: GateResult | None, universe: list[MarketSpec]
) -> list[DecisionRow]:
    choices = {item.symbol: item for item in decision.decisions} if decision is not None else {}
    gated = {item.symbol: item for item in gate.targets} if gate is not None else {}
    return [
        _decision_row(spec.display_name, choices.get(spec.symbol), gated.get(spec.symbol))
        for spec in universe
    ]


def _decision_row(name: str, choice: MarketDecision | None, gated: GatedTarget | None) -> DecisionRow:
    requested = choice.target_weight if choice is not None else None
    if requested is None and gated is not None:
        requested = gated.requested_weight
    approved = gated.approved_weight if gated is not None else None
    reasons = ", ".join(gated.reasons) if gated is not None and gated.reasons else "none"
    probability = f"{choice.prob_up_24h:.2f}" if choice is not None else "n/a"
    return DecisionRow(
        market=name,
        requested=_weight(requested),
        approved=_weight(approved),
        status=gated.status if gated is not None else "n/a",
        reasons=reasons,
        probability=probability,
        thesis=choice.thesis if choice is not None else "",
        requested_class=_tone(requested),
        approved_class=_tone(approved),
    )


def _snapshot_rows(snapshot: Snapshot, universe: list[MarketSpec]) -> list[SnapshotRow]:
    names = {item.symbol: item for item in universe}
    rows: list[SnapshotRow] = []
    for market in snapshot.markets:
        spec = names.get(market.symbol)
        asset = _CLASS_LABELS[spec.asset_class] if spec is not None else _CLASS_LABELS[market.asset_class]
        label = spec.display_name if spec is not None else market.display_name
        rows.append(
            SnapshotRow(
                market=label,
                asset_class=asset,
                status="Open" if market.market_open else "Closed",
                mark=_price(market.mark),
                ret_24h=_pct(market.ret_24h),
                ret_7d=_pct(market.ret_7d),
                ret_30d=_pct(market.ret_30d),
                vol=_plain_pct(market.vol_30d_ann),
                funding=_pct(market.funding_ann),
                open_interest=_dollars(market.open_interest_usd),
                volume=_dollars(market.volume_24h_usd),
                ret_24h_class=_tone(market.ret_24h),
                ret_7d_class=_tone(market.ret_7d),
                ret_30d_class=_tone(market.ret_30d),
                funding_class=_tone(market.funding_ann),
            )
        )
    return rows


def _order_row(item: OrderOutcome) -> OrderRow:
    return OrderRow(
        market=item.symbol,
        side=item.side,
        requested=_price(item.requested_size),
        filled=_price(item.filled_size),
        price=_price(item.avg_px) if item.avg_px is not None else "n/a",
        fee=_usd(item.fee_usd),
        status=item.status,
    )


def _cost_rows(record: CycleRecord) -> tuple[list[CostRow], str]:
    rows: list[CostRow] = []
    amounts: list[float] = []
    if record.news_usage is not None:
        rows.append(_usage_row(record.news_usage))
        amounts.append(record.news_usage.cost_usd)
    for run in record.runs:
        for usage in run.usage:
            rows.append(_usage_row(usage))
            amounts.append(usage.cost_usd)
    return rows, _usd(sum(amounts))


def _usage_row(usage: LlmUsage) -> CostRow:
    return CostRow(
        purpose=usage.purpose,
        model=usage.model_id,
        input_tokens=f"{usage.input_tokens:,}",
        output_tokens=f"{usage.output_tokens:,}",
        cache_write=f"{usage.cache_write_tokens:,}",
        cache_read=f"{usage.cache_read_tokens:,}",
        searches=str(usage.web_searches),
        cost=_usd(usage.cost_usd),
    )


def _write_data(
    out_dir: Path, moment: datetime, scores: ScoreReport | None, equity: list[EquityPoint]
) -> None:
    payload = {
        "generated_at": moment.isoformat(),
        "scores": None if scores is None else scores.model_dump(mode="json"),
        "equity": [item.model_dump(mode="json") for item in equity],
    }
    (out_dir / "data.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_transcript(path: str | None) -> str:
    if not path:
        return ""
    file = Path(path)
    if not file.is_file():
        return ""
    try:
        payload = _load_payload(file)
    except (OSError, ValueError, EOFError, UnicodeError, zlib.error):
        logger.warning("Skipping unreadable transcript at %s", path)
        return ""
    return "\n\n".join(_thinking(payload))


def _load_payload(file: Path) -> object:
    if file.name.endswith(".gz"):
        with gzip.open(file, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(file.read_text(encoding="utf-8"))


def _thinking(node: object) -> list[str]:
    found: list[str] = []
    _walk(node, found)
    return found


def _walk(node: object, found: list[str]) -> None:
    if isinstance(node, dict):
        mapping = cast("dict[object, object]", node)
        if mapping.get("type") == "thinking":
            raw = mapping.get("thinking")
            if not isinstance(raw, str):
                raw = mapping.get("text")
            if isinstance(raw, str) and raw.strip():
                found.append(raw.strip())
        for value in mapping.values():
            _walk(value, found)
        return
    if isinstance(node, list):
        for item in cast("list[object]", node):
            _walk(item, found)


def _grouped(points: list[EquityPoint]) -> dict[str, list[EquityPoint]]:
    grouped: dict[str, list[EquityPoint]] = {}
    for point in points:
        grouped.setdefault(point.portfolio_id, []).append(point)
    return grouped


def _indexed(series: list[EquityPoint], times: list[datetime]) -> list[float | None]:
    base = series[0].equity
    by_time = {item.at: item.equity for item in series}
    values: list[float | None] = []
    for moment in times:
        point = by_time.get(moment)
        if point is None or base == 0:
            values.append(None)
        else:
            values.append(round(point / base * 100, 2))
    return values


def _series_return(points: list[EquityPoint]) -> float | None:
    if len(points) < 2 or points[0].equity == 0:
        return None
    return points[-1].equity / points[0].equity - 1


def _account_equity(account: PaperAccount) -> float:
    total = account.cash
    for position in account.positions.values():
        mark = account.last_marks.get(position.symbol, position.entry_px)
        total += position.size * (mark - position.entry_px)
    return total


def _unrealised(position: PaperPosition, mark: float) -> float | None:
    if position.entry_px == 0:
        return None
    sign = 1.0 if position.size > 0 else -1.0
    return sign * (mark / position.entry_px - 1)


def _source(source: str) -> SourceLink:
    text = source.strip()
    lowered = text.lower()
    if lowered.startswith("https://") or lowered.startswith("http://"):
        return SourceLink(href=text, label=text)
    return SourceLink(href=None, label=text)


def _session_line(snapshot: Snapshot) -> str:
    state = "open" if snapshot.session.tradfi_open else "closed"
    weekend = "yes" if snapshot.session.pre_weekend else "no"
    return f"US market session: {state}. Pre weekend window: {weekend}."


def _summary(run: ConfigRun | None) -> str:
    if run is None or run.decision is None:
        return "No decision recorded"
    text = " ".join(run.decision.market_view.split())
    if len(text) <= 140:
        return text
    return text[:137] + "..."


def _order_count(run: ConfigRun | None) -> int:
    if run is None:
        return 0
    return sum(1 for item in run.outcomes if item.status != "skipped")


def _cycle_cost(record: CycleRecord) -> float:
    total = record.news_usage.cost_usd if record.news_usage is not None else 0.0
    for run in record.runs:
        total += sum(item.cost_usd for item in run.usage)
    return total


def _llm_total(cycles: list[CycleRecord]) -> float:
    return sum(_cycle_cost(record) for record in cycles)


def _theses(run: ConfigRun | None) -> dict[str, str]:
    if run is None or run.decision is None:
        return {}
    return {item.symbol: item.thesis for item in run.decision.decisions}


def _run(record: CycleRecord, config_id: str) -> ConfigRun | None:
    for run in record.runs:
        if run.config_id == config_id:
            return run
    return None


def _portfolio_score(report: ScoreReport | None, portfolio_id: str) -> PortfolioScore | None:
    if report is None:
        return None
    for item in report.portfolios:
        if item.portfolio_id == portfolio_id:
            return item
    return None


def _forecast_score(report: ScoreReport | None, config_id: str) -> ForecastScore | None:
    if report is None:
        return None
    for item in report.forecasts:
        if item.config_id == config_id:
            return item
    return None


def _champion_ids(config: FundConfig) -> tuple[str, str]:
    config_id = config.champion.config_id
    return config_id, f"config:{config_id}"


def _role(portfolio_id: str, champion_portfolio_id: str) -> str:
    if portfolio_id == champion_portfolio_id:
        return "champion"
    if portfolio_id.startswith("baseline:"):
        return "baseline"
    return "other"


def _role_rank(portfolio_id: str, champion_portfolio_id: str) -> int:
    role = _role(portfolio_id, champion_portfolio_id)
    if role == "champion":
        return 2
    if role == "baseline":
        return 1
    return 0


def _series_label(portfolio_id: str, champion_portfolio_id: str) -> str:
    if portfolio_id == champion_portfolio_id:
        return f"{portfolio_id} (champion)"
    return portfolio_id


def _halt_text(state: dict[str, str] | None) -> str | None:
    if state is None:
        return None
    reason = state.get("reason")
    if reason:
        return reason
    return ", ".join(state.values()) or "A halt is active"


def _slug(cycle_id: str) -> str:
    return cycle_id.replace("/", "__").replace(":", "__")


def _aware(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone aware")
    return now.astimezone(UTC)


def _when(moment: datetime) -> str:
    utc = moment.astimezone(UTC)
    month = _MONTHS[utc.month - 1]
    return f"{utc.day:02d} {month} {utc.year}, {utc.hour:02d}:{utc.minute:02d} UTC"


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.2f}%"


def _plain_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _signed(value: float | None, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def _digits(value: float | None, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _usd(value: float) -> str:
    return f"${value:,.2f}"


def _dollars(value: float) -> str:
    return f"${value:,.0f}"


def _price(value: float) -> str:
    return f"{value:,.2f}"


def _weight(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}"


def _tone(value: float | None) -> str:
    if value is None or value == 0:
        return ""
    if value > 0:
        return "positive"
    return "negative"
