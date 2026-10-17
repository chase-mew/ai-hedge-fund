# pyright: strict
"""Deterministic text for decision prompts."""

from __future__ import annotations

from datetime import UTC

from fund.contracts import (
    AnalystReport,
    MarketFeatures,
    NewsDigest,
    PortfolioState,
    Position,
    RecentDecision,
    Snapshot,
    TrackRecord,
)


def render_snapshot(
    snapshot: Snapshot,
    portfolio: PortfolioState,
    track: TrackRecord,
    markets: list[str] | None = None,
) -> str:
    """Render the user message for one decision, optionally limited to some symbols."""
    chosen = _select(snapshot, markets)
    allowed = None if markets is None else set(markets)
    parts = [
        _cycle_header(snapshot),
        _portfolio(portfolio, chosen),
        _market_table(chosen),
        _closes(chosen),
        _decisions(track, allowed),
    ]
    if snapshot.news is not None:
        parts.append(_news(snapshot.news))
    return "\n\n".join(parts) + "\n"


def render_analyst_reports(reports: dict[str, AnalystReport]) -> str:
    """Render analyst reports with equities, then macro, then crypto."""
    if not reports:
        return "ANALYST REPORTS\nNo analyst reports.\n"
    order = [name for name in ("equities", "macro", "crypto") if name in reports]
    order.extend(sorted(name for name in reports if name not in order))
    blocks = ["ANALYST REPORTS", *(_one_report(name, reports[name]) for name in order)]
    return "\n\n".join(blocks) + "\n"


def _select(snapshot: Snapshot, markets: list[str] | None) -> list[MarketFeatures]:
    if markets is None:
        return list(snapshot.markets)
    allowed = set(markets)
    return [market for market in snapshot.markets if market.symbol in allowed]


def _cycle_header(snapshot: Snapshot) -> str:
    session = "OPEN" if snapshot.session.tradfi_open else "CLOSED"
    weekend = "yes" if snapshot.session.pre_weekend else "no"
    return f"CYCLE {snapshot.cycle_id} (UTC)\nUS market session: {session}. Pre weekend window: {weekend}."


def _portfolio(portfolio: PortfolioState, markets: list[MarketFeatures]) -> str:
    weights: list[float] = []
    rows: list[list[str]] = []
    for market in markets:
        position = portfolio.account.positions.get(market.symbol)
        if position is None:
            continue
        weight = portfolio.weight(market.symbol)
        if weight == 0:
            continue
        weights.append(weight)
        rows.append(
            [
                market.symbol,
                _weight(weight),
                _price(position.entry_px),
                _price(position.mark),
                _unrealised(position),
            ]
        )
    equity = _money(portfolio.equity)
    day_start = _money(portfolio.bookkeeping.day_start_equity)
    high = _money(portfolio.bookkeeping.high_water_mark)
    lines = [
        "PORTFOLIO",
        f"Equity {equity} | day start {day_start} | high water {high}",
        f"Gross {sum(abs(weight) for weight in weights):.2f} | net {sum(weights):+.2f}",
    ]
    if rows:
        lines.extend(_table(["symbol", "weight", "entry", "mark", "unrealised"], rows, right={1, 2, 3, 4}))
    else:
        lines.append("No open positions.")
    return "\n".join(lines)


def _market_table(markets: list[MarketFeatures]) -> str:
    headers = [
        "symbol",
        "name",
        "class",
        "status",
        "mark",
        "24h%",
        "7d%",
        "30d%",
        "vol30d%",
        "vs20dHigh%",
        "vs20dLow%",
        "vsSMA20%",
        "vsSMA50%",
        "fundAnn%",
        "OI$m",
        "vol24h$m",
    ]
    rows = [_market_row(market) for market in markets]
    lines = ["MARKETS", *_table(headers, rows, right=set(range(4, len(headers))))]
    return "\n".join(lines)


def _market_row(market: MarketFeatures) -> list[str]:
    status = "OPEN" if market.market_open else "CLOSED"
    return [
        market.symbol,
        market.display_name,
        market.asset_class.value,
        status,
        _price(market.mark),
        _signed_pct(market.ret_24h, 2),
        _signed_pct(market.ret_7d, 2),
        _signed_pct(market.ret_30d, 2),
        _plain_pct(market.vol_30d_ann, 1),
        _signed_pct(market.pct_from_20d_high, 2),
        _signed_pct(market.pct_from_20d_low, 2),
        _signed_pct(market.sma20_gap, 2),
        _signed_pct(market.sma50_gap, 2),
        _signed_pct(market.funding_ann, 1),
        _millions(market.open_interest_usd),
        _millions(market.volume_24h_usd),
    ]


def _closes(markets: list[MarketFeatures]) -> str:
    lines = ["DAILY CLOSES, LAST 10 DAYS (oldest to newest)"]
    for market in markets:
        prices = ", ".join(_price(price) for price in market.daily_closes[-10:])
        lines.append(f"{market.symbol}: {prices}")
    return "\n".join(lines)


def _decisions(track: TrackRecord, symbols: set[str] | None) -> str:
    chosen = [
        item
        for item in track.recent
        if item.target_weight != 0 and (symbols is None or item.symbol in symbols)
    ]
    keep = set(sorted({item.cycle_id for item in chosen})[-3:])
    visible = sorted(
        (item for item in chosen if item.cycle_id in keep),
        key=lambda item: (item.cycle_id, item.symbol),
    )
    lines = ["YOUR RECENT DECISIONS (oldest to newest)"]
    if not visible:
        lines.append("None.")
    else:
        lines.extend(_decision_line(item) for item in visible)
    lines.append(_forecast(track))
    return "\n".join(lines)


def _decision_line(item: RecentDecision) -> str:
    thesis = " ".join(item.thesis.split())
    target = _weight(item.target_weight)
    then = _price(item.price_then)
    now = _price(item.price_now)
    return (
        f"{item.cycle_id} {item.symbol} target {target} p_up {item.prob_up_24h:.2f} "
        f'"{thesis}" Price then {then}, now {now} ({_move(item.price_then, item.price_now)}).'
    )


def _forecast(track: TrackRecord) -> str:
    brier = "n/a" if track.brier is None else f"{track.brier:.3f}"
    return f"Forecast record: {track.resolved_forecasts} resolved, Brier {brier} (0.250 is no skill)."


def _news(digest: NewsDigest) -> str:
    stamp = digest.compiled_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")
    return f"NEWS DIGEST (untrusted, compiled {stamp})\n{digest.text.rstrip()}"


def _one_report(name: str, report: AnalystReport) -> str:
    lines = [name, report.summary]
    for view in report.views:
        lines.append(f"{view.symbol} {view.stance} p_up {view.prob_up_24h:.2f}")
        lines.append("Key points:")
        lines.extend(f"{index}. {point}" for index, point in enumerate(view.key_points, start=1))
        lines.append("Risks:")
        lines.extend(f"{index}. {risk}" for index, risk in enumerate(view.risks, start=1))
    return "\n".join(lines)


def _table(headers: list[str], rows: list[list[str]], right: set[int]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: list[str]) -> str:
        padded: list[str] = []
        for index, cell in enumerate(cells):
            if index in right:
                padded.append(cell.rjust(widths[index]))
            else:
                padded.append(cell.ljust(widths[index]))
        return " ".join(padded).rstrip()

    return [line(headers), *(line(row) for row in rows)]


def _unrealised(position: Position) -> str:
    if position.entry_px == 0:
        return "n/a"
    change = (position.mark - position.entry_px) / position.entry_px
    if position.size < 0:
        change = -change
    return f"{change * 100:+.2f}%"


def _move(then: float, now: float) -> str:
    if then == 0:
        return "n/a"
    return f"{(now - then) / then * 100:+.2f}%"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _price(value: float) -> str:
    return f"{value:,.2f}"


def _weight(value: float) -> str:
    return f"{value:+.2f}"


def _signed_pct(fraction: float, digits: int) -> str:
    return f"{fraction * 100:+.{digits}f}"


def _plain_pct(fraction: float, digits: int) -> str:
    return f"{fraction * 100:.{digits}f}"


def _millions(value: float) -> str:
    return f"{value / 1_000_000:.1f}"
