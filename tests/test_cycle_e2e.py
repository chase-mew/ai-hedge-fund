"""Offline cycle tests. Hourly bars are expanded here because the fixtures store daily candles."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from typer.testing import CliRunner

from fund.brain.fake import FakeClaude
from fund.brain.llm import LlmResult
from fund.cli import app
from fund.config import Secrets, load_config
from fund.contracts import (
    AccountSnapshot,
    Candle,
    CycleRecord,
    LlmUsage,
    MarketMeta,
    OrderOutcome,
    PortfolioDecision,
    Position,
    Side,
    StopOrder,
)
from fund.cycle import CycleContext, run_cycle
from fund.journal.store import Journal
from tests.conftest import FixtureData
from tests.factories import make_decision

NOW = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)
runner = CliRunner()


class HourlyData(FixtureData):
    """Recorded daily data, plus flat hourly bars so stops and resolutions have a clock."""

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime) -> list[Candle]:
        if interval != "1h":
            return super().candles(symbol, interval, start, end)
        mark = self.quotes([symbol])[symbol].mark
        cursor = start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        stop = end.astimezone(UTC)
        rows: list[Candle] = []
        while cursor <= stop and len(rows) < 80:
            rows.append(
                Candle(open_time=cursor, open=mark, high=mark, low=mark, close=mark * 1.01, volume=0.0)
            )
            cursor += timedelta(hours=1)
        return rows


class RecordingVenue:
    """Stand in that records orders and stops."""

    name = "recording"

    def __init__(self, equity: float, marks: dict[str, float], meta: dict[str, MarketMeta]) -> None:
        self._equity = equity
        self._marks = marks
        self._meta = meta
        self._positions: dict[str, Position] = {}
        self.orders: list[tuple[str, Side, float]] = []
        self.stops: list[tuple[str, StopOrder | None]] = []

    def market_meta(self) -> dict[str, MarketMeta]:
        return self._meta

    def marks(self) -> dict[str, float]:
        return self._marks

    def account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=self._equity, positions=dict(self._positions))

    def market_order(self, symbol: str, side: Side, size: float, reduce_only: bool) -> OrderOutcome:
        del reduce_only
        self.orders.append((symbol, side, size))
        mark = self._marks.get(symbol, 1.0)
        signed = size if side == "buy" else -size
        self._positions[symbol] = Position(symbol=symbol, size=signed, entry_px=mark, mark=mark)
        return OrderOutcome(
            symbol=symbol,
            side=side,
            requested_size=size,
            filled_size=size,
            avg_px=mark,
            fee_usd=0.0,
            status="filled",
            order_id=f"rec-{len(self.orders)}",
        )

    def close_position(self, symbol: str) -> OrderOutcome:
        self.orders.append((symbol, "sell", 0.0))
        self._positions.pop(symbol, None)
        return OrderOutcome(
            symbol=symbol,
            side="sell",
            requested_size=0.0,
            filled_size=0.0,
            avg_px=None,
            fee_usd=0.0,
            status="skipped",
        )

    def ensure_leverage(self, symbol: str, leverage: int, cross: bool) -> None:
        del symbol, leverage, cross

    def sync_stop(self, symbol: str, stop: StopOrder | None) -> None:
        self.stops.append((symbol, stop))

    def fees_since(self, start: datetime) -> dict[str, float]:
        del start
        return {}


class _FixedClaude:
    def __init__(self, decision: PortfolioDecision) -> None:
        self._decision = decision

    async def parse[T](
        self,
        *,
        purpose: str,
        model: object,
        system: str,
        user: str,
        output_type: type[T],
    ) -> LlmResult[T]:
        del purpose, model, system, user, output_type
        output = LlmResult(self._decision, _free("decision"), {"request": {}, "response": {}}, None)
        return cast(LlmResult[T], output)

    async def search(
        self,
        *,
        purpose: str,
        model: object,
        system: str,
        user: str,
        news: object,
    ) -> LlmResult[str]:
        del purpose, model, system, user, news
        return LlmResult("Nothing notable.", _free("news"), {"request": {}, "response": {}}, None)


def test_paper_cycle_journals_configs_baselines_and_forecasts(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    record = _paper(journal, NOW)
    config = load_config()
    assert record.errors == []
    assert journal.has_cycle(record.cycle_id)
    ids = [f"config:{item.config_id}" for item in config.configs]
    ids.extend(f"baseline:{name}" for name in ("sp500", "equal_weight", "trend", "random"))
    for portfolio_id in ids:
        assert journal.load_paper(portfolio_id) is not None
    points = journal.load_equity()
    assert {point.portfolio_id for point in points} == set(ids)
    forecasts = _count_jsonl(tmp_path / "forecasts.jsonl")
    assert forecasts == len(config.configs) * len(config.universe)
    for item in config.configs:
        slug = item.config_id.replace("/", "__").replace(":", "__")
        matches = list(tmp_path.glob(f"transcripts/**/{slug}.json.gz"))
        assert matches


def test_same_slot_is_a_noop_and_the_next_slot_does_not_resolve(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    first = _paper(journal, NOW)
    again = _paper(journal, NOW)
    assert again.cycle_id == first.cycle_id
    assert len(journal.load_cycles()) == 1
    assert len(journal.load_equity()) == 8
    _paper(journal, NOW + timedelta(hours=4))
    assert len(journal.load_cycles()) == 2
    assert _count_jsonl(tmp_path / "resolutions.jsonl") == 0


def test_forecasts_resolve_once_the_hour_after_the_horizon_has_closed(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    _paper(journal, NOW)
    _paper(journal, NOW + timedelta(hours=24))
    assert _count_jsonl(tmp_path / "resolutions.jsonl") == 0
    _paper(journal, NOW + timedelta(hours=28))
    expected = len(load_config().configs) * len(load_config().universe)
    assert _count_jsonl(tmp_path / "resolutions.jsonl") == expected


def test_halt_stops_the_next_cycle(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    _paper(journal, NOW)
    before = len(journal.load_equity())
    journal.set_halt("manual", NOW)
    later = NOW + timedelta(hours=8)
    record = _paper(journal, later)
    assert record.errors == ["halted"]
    assert record.runs == []
    assert len(journal.load_equity()) == before


def test_live_orders_reach_the_recording_venue(tmp_path: Path) -> None:
    venue = _live(tmp_path, equity=500.0, autonomy="auto", approve=None)
    assert venue.orders
    assert any(stop is not None for _, stop in venue.stops)
    assert Journal(tmp_path).load_live_bookkeeping() is not None


def test_live_equity_above_the_ceiling_sends_nothing(tmp_path: Path) -> None:
    venue = _live(tmp_path, equity=5_000.0, autonomy="auto", approve=None)
    record = Journal(tmp_path).load_cycles()[-1]
    assert venue.orders == []
    assert any("ceiling" in item for item in record.errors)


def test_declined_approval_sends_nothing(tmp_path: Path) -> None:
    venue = _live(tmp_path, equity=500.0, autonomy="approve", approve=lambda _gate, _plans: False)
    assert venue.orders == []


def test_cli_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("run", "status", "score", "dashboard", "halt", "resume", "doctor", "replay"):
        assert name in result.stdout


def test_status_score_and_dashboard_read_an_empty_journal(tmp_path: Path) -> None:
    path = _config_copy(tmp_path)
    status = runner.invoke(app, ["status", "--config", str(path)])
    assert status.exit_code == 0
    assert "Last cycle: none." in status.stdout
    scored = runner.invoke(app, ["score", "--config", str(path)])
    assert scored.exit_code == 0
    site = runner.invoke(app, ["dashboard", "--config", str(path), "--out", str(tmp_path / "site")])
    assert site.exit_code == 0
    assert (tmp_path / "site" / "index.html").is_file()
    assert "No cycles yet" in (tmp_path / "site" / "index.html").read_text()


def _paper(journal: Journal, now: datetime) -> CycleRecord:
    record = run_cycle(
        _context(journal, now, live=None, autonomy="approve", claude=FakeClaude(), mode="paper")
    )
    assert isinstance(record, CycleRecord)
    return record


def _live(
    root: Path,
    equity: float,
    autonomy: str,
    approve: object,
) -> RecordingVenue:
    data = HourlyData()
    config = load_config()
    marks = {spec.symbol: data.quotes([spec.symbol])[spec.symbol].mark for spec in config.universe}
    meta = data.meta([spec.symbol for spec in config.universe])
    venue = RecordingVenue(equity, marks, meta)
    decision = make_decision(weights={"BTC": 0.10}, probs={"BTC": 0.62})
    run_cycle(
        _context(
            Journal(root),
            NOW,
            live=venue,
            autonomy=autonomy,
            claude=_FixedClaude(decision),
            mode="testnet",
            approve=approve,
        )
    )
    return venue


def _context(
    journal: Journal,
    now: datetime,
    live: RecordingVenue | None,
    autonomy: str,
    claude: object,
    mode: str,
    approve: object = None,
) -> CycleContext:
    from typing import cast

    from fund.brain.llm import Llm
    from fund.contracts import Mode
    from fund.venues.base import Venue

    config = load_config()
    fund = config.fund.model_copy(update={"autonomy": autonomy})
    chosen: Mode = "paper" if mode == "paper" else "testnet"
    return CycleContext(
        config=config.model_copy(update={"fund": fund}),
        secrets=Secrets(),
        journal=journal,
        data=HourlyData(),
        claude=cast(Llm, claude),
        mode=chosen,
        now=now,
        live_venue=cast(Venue, live) if live is not None else None,
        approve=approve,
    )


def _config_copy(tmp_path: Path) -> Path:
    source = Path("fund.toml").read_text()
    text = source.replace('journal_dir = "journal"', f'journal_dir = "{tmp_path.as_posix()}/journal"')
    path = tmp_path / "fund.toml"
    path.write_text(text)
    return path


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _free(purpose: str) -> LlmUsage:
    return LlmUsage(
        purpose=purpose,
        model_id="fake",
        input_tokens=0,
        output_tokens=0,
        cache_write_tokens=0,
        cache_read_tokens=0,
        web_searches=0,
        cost_usd=0.0,
    )
