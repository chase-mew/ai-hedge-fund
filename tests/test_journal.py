"""Journal files, bookkeeping and forecast resolution."""

import gzip
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from fund.contracts import (
    AccountSnapshot,
    Bookkeeping,
    Candle,
    ConfigRun,
    CycleRecord,
    EquityPoint,
    Forecast,
    LlmUsage,
    MarketDecision,
    PaperAccount,
    PaperPosition,
    PortfolioDecision,
    PortfolioScore,
    Position,
    Resolution,
    ScoreReport,
    Snapshot,
    StopOrder,
)
from fund.journal.store import Journal, record_flips, resolve, update_bookkeeping
from tests.factories import make_snapshot

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)


def _at(cycle_id: str) -> datetime:
    return datetime.fromisoformat(cycle_id.replace("Z", "+00:00"))


def _usage(cost: float) -> LlmUsage:
    return LlmUsage(
        purpose="decide",
        model_id="claude-opus-5-5",
        input_tokens=1,
        output_tokens=1,
        cache_write_tokens=0,
        cache_read_tokens=0,
        web_searches=0,
        cost_usd=cost,
    )


def _decision(rows: list[tuple[str, float, float, str]]) -> PortfolioDecision:
    return PortfolioDecision(
        market_view="Fixture view.",
        decisions=[
            MarketDecision(
                symbol=symbol,
                thesis=thesis,
                invalidation="A close through the average.",
                prob_up_24h=prob,
                target_weight=weight,
                stop_loss_pct=5.0,
            )
            for symbol, weight, prob, thesis in rows
        ],
    )


def _run(config_id: str, decision: PortfolioDecision | None, costs: list[float]) -> ConfigRun:
    return ConfigRun(
        config_id=config_id,
        portfolio_id=f"config:{config_id}",
        decision=decision,
        error=None,
        gate=None,
        outcomes=[],
        usage=[_usage(cost) for cost in costs],
        transcript_path=None,
        equity_before=10000.0,
        equity_after=10000.0,
    )


def _cycle(
    cycle_id: str,
    started: datetime,
    runs: list[ConfigRun] | None = None,
    news_cost: float | None = None,
    snapshot: Snapshot | None = None,
) -> CycleRecord:
    return CycleRecord(
        cycle_id=cycle_id,
        mode="paper",
        started_at=started,
        finished_at=started,
        snapshot=snapshot or make_snapshot(now=started, cycle_id=cycle_id, as_of=started),
        news_usage=None if news_cost is None else _usage(news_cost),
        runs=runs or [],
        errors=[],
    )


def _forecast(
    forecast_id: str = "f1",
    *,
    config_id: str = "opus-5.5/single",
    resolves_at: datetime | None = None,
    ref_price: float = 100.0,
    prob_up: float = 0.6,
) -> Forecast:
    made = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    return Forecast(
        forecast_id=forecast_id,
        cycle_id="2026-10-16T12:00Z",
        config_id=config_id,
        symbol="BTC",
        made_at=made,
        resolves_at=resolves_at or (made + timedelta(hours=24)),
        prob_up=prob_up,
        ref_price=ref_price,
    )


def _candle(open_time: datetime, close: float) -> Candle:
    return Candle(open_time=open_time, open=close, high=close, low=close, close=close, volume=1.0)


def _account(sizes: dict[str, float]) -> AccountSnapshot:
    positions = {
        symbol: Position(symbol=symbol, size=size, entry_px=10.0, mark=10.0)
        for symbol, size in sizes.items()
        if size != 0.0
    }
    return AccountSnapshot(equity=10000.0, positions=positions)


def _with_marks(base: Snapshot, marks: dict[str, float], cycle_id: str, moment: datetime) -> Snapshot:
    markets = [
        market.model_copy(update={"mark": marks[market.symbol]}) if market.symbol in marks else market
        for market in base.markets
    ]
    return base.model_copy(update={"markets": markets, "cycle_id": cycle_id, "as_of": moment})


def test_cycle_round_trip_and_has_cycle(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    cycle_id = "2026-10-17T12:00Z"
    assert journal.has_cycle(cycle_id) is False
    record = _cycle(cycle_id, _at(cycle_id), news_cost=0.2)
    path = journal.write_cycle(record)
    assert path == tmp_path / "cycles" / "2026" / "10" / "17" / "1200.json"
    assert path.read_text() == record.model_dump_json()
    assert journal.has_cycle(cycle_id) is True
    assert journal.load_cycles() == [record]
    replacement = record.model_copy(update={"errors": ["replaced"]})
    journal.write_cycle(replacement)
    assert journal.load_cycles() == [replacement]


def test_load_cycles_orders_oldest_first_and_limits_to_newest(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    assert journal.load_cycles() == []
    for cycle_id in ("2026-10-17T16:00Z", "2026-10-17T08:00Z", "2026-10-17T12:00Z"):
        journal.write_cycle(_cycle(cycle_id, _at(cycle_id)))
    ordered = ["2026-10-17T08:00Z", "2026-10-17T12:00Z", "2026-10-17T16:00Z"]
    assert [row.cycle_id for row in journal.load_cycles()] == ordered
    assert [row.cycle_id for row in journal.load_cycles(limit=2)] == ordered[-2:]
    assert [row.cycle_id for row in journal.load_cycles(limit=10)] == ordered
    assert journal.load_cycles(limit=0) == []


def test_writes_replace_a_temp_file_in_the_same_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        source = Path(src)
        target = Path(dst)
        assert source.parent == target.parent
        assert source != target
        assert source.suffix == ".tmp"
        assert source.is_file()
        calls.append((source, target))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    journal = Journal(tmp_path)
    journal.write_cycle(_cycle("2026-10-17T12:00Z", NOW))
    journal.append_forecasts([_forecast()])
    journal.save_paper(
        PaperAccount(
            portfolio_id="config:opus-5.5/single",
            cash=10000.0,
            positions={},
            last_marks={},
            updated_at=NOW,
            bookkeeping=Bookkeeping(day="2026-10-17", day_start_equity=10000.0, high_water_mark=10000.0),
        )
    )
    written = {target.relative_to(tmp_path).as_posix() for _, target in calls}
    assert "cycles/2026/10/17/1200.json" in written
    assert "forecasts.jsonl" in written
    assert "state/paper/config__opus-5.5__single.json" in written
    assert list(tmp_path.rglob("*.tmp")) == []


def test_failed_replace_removes_the_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    journal = Journal(tmp_path)
    with pytest.raises(OSError, match="replace failed"):
        journal.write_cycle(_cycle("2026-10-17T12:00Z", NOW))
    assert list(tmp_path.rglob("*.tmp")) == []
    assert journal.has_cycle("2026-10-17T12:00Z") is False


def test_transcript_round_trip_uses_the_config_slug(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    payload = [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "world"}]
    relative = journal.write_transcripts("2026-10-19T12:00Z", "opus-5.5/single", payload)
    assert relative == "transcripts/2026/10/19/1200/opus-5.5__single.json.gz"
    restored = json.loads(gzip.decompress((tmp_path / relative).read_bytes()))
    assert restored == payload


def test_jsonl_files_round_trip(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    assert journal.load_equity() == []
    first = _forecast("f1")
    second = _forecast("f2", prob_up=0.2)
    journal.append_forecasts([first])
    journal.append_forecasts([second])
    forecast_text = first.model_dump_json() + "\n" + second.model_dump_json() + "\n"
    assert (tmp_path / "forecasts.jsonl").read_text() == forecast_text
    resolution = Resolution(forecast_id="f1", resolved_at=NOW, price=110.0, outcome=True)
    journal.append_resolutions([resolution])
    assert (tmp_path / "resolutions.jsonl").read_text() == resolution.model_dump_json() + "\n"
    early = EquityPoint(
        at=NOW,
        cycle_id="2026-10-17T12:00Z",
        portfolio_id="live",
        equity=100.0,
        gross=0.1,
        net=0.1,
        fees_cum=1.0,
        funding_cum=0.2,
        llm_cost_cum=0.3,
    )
    later = early.model_copy(
        update={"at": NOW + timedelta(hours=4), "cycle_id": "2026-10-17T16:00Z", "equity": 110.0}
    )
    journal.append_equity([early])
    journal.append_equity([later])
    assert journal.load_equity() == [early, later]


def test_append_repairs_a_missing_newline(tmp_path: Path) -> None:
    first = _forecast("f1")
    path = tmp_path / "forecasts.jsonl"
    path.write_bytes(first.model_dump_json().encode())
    second = _forecast("f2")
    Journal(tmp_path).append_forecasts([second])
    rows = [Forecast.model_validate_json(line) for line in path.read_text().splitlines()]
    assert rows == [first, second]


def test_empty_append_does_not_create_a_file(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append_forecasts([])
    journal.append_resolutions([])
    journal.append_equity([])
    assert not (tmp_path / "forecasts.jsonl").exists()
    assert not (tmp_path / "resolutions.jsonl").exists()
    assert not (tmp_path / "equity.jsonl").exists()


def test_paper_account_round_trip(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    account = PaperAccount(
        portfolio_id="config:opus-5.5/single",
        cash=9900.0,
        positions={
            "BTC": PaperPosition(
                symbol="BTC",
                size=0.01,
                entry_px=100.0,
                stop=StopOrder(symbol="BTC", side="sell", size=0.01, trigger_px=90.0),
            )
        },
        last_marks={"BTC": 101.0},
        updated_at=NOW,
        bookkeeping=Bookkeeping(
            day="2026-10-17",
            day_start_equity=10000.0,
            high_water_mark=10100.0,
            last_flip_at={"BTC": NOW},
        ),
        fees_cum=1.5,
        funding_cum=0.25,
        llm_cost_cum=0.4,
    )
    assert journal.load_paper(account.portfolio_id) is None
    journal.save_paper(account)
    path = tmp_path / "state" / "paper" / "config__opus-5.5__single.json"
    assert path.read_text() == account.model_dump_json()
    assert journal.load_paper(account.portfolio_id) == account


def test_live_bookkeeping_round_trip(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    book = Bookkeeping(
        day="2026-10-17",
        day_start_equity=10000.0,
        high_water_mark=11000.0,
        last_flip_at={"ETH": NOW},
    )
    assert journal.load_live_bookkeeping() is None
    journal.save_live_bookkeeping(book)
    assert (tmp_path / "state" / "live.json").read_text() == book.model_dump_json()
    assert journal.load_live_bookkeeping() == book


def test_halt_set_and_clear(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    assert journal.halt_state() is None
    journal.clear_halt()
    journal.set_halt("drawdown", NOW)
    assert journal.halt_state() == {"reason": "drawdown", "at": NOW.isoformat()}
    later = NOW + timedelta(hours=1)
    journal.set_halt("manual", later)
    assert journal.halt_state() == {"reason": "manual", "at": later.isoformat()}
    journal.clear_halt()
    assert journal.halt_state() is None
    assert not (tmp_path / "state" / "halt.json").exists()


def test_scores_round_trip(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    report = ScoreReport(
        generated_at=NOW,
        portfolios=[
            PortfolioScore(
                portfolio_id="live",
                kind="live",
                cycles=1,
                start_equity=10000.0,
                equity=10010.0,
                total_return=0.001,
                return_net_of_llm=0.001,
                ann_vol=None,
                sharpe=None,
                sortino=None,
                max_drawdown=0.0,
                fees=1.0,
                funding=0.0,
                llm_cost=0.2,
            )
        ],
        forecasts=[],
    )
    assert journal.load_scores() is None
    journal.write_scores(report)
    assert (tmp_path / "scores.json").read_text() == report.model_dump_json()
    assert journal.load_scores() == report


def test_llm_spend_on_sums_the_utc_day(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    early = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
    late_start = datetime(2026, 10, 18, 0, 30, tzinfo=UTC)
    journal.write_cycle(
        _cycle("2026-10-17T12:00Z", early, runs=[_run("opus-5.5/single", None, [0.3, 0.05])], news_cost=0.2)
    )
    journal.write_cycle(_cycle("2026-10-17T16:00Z", early, runs=[_run("haiku-5.5/single", None, [0.1])]))
    journal.write_cycle(
        _cycle("2026-10-17T20:00Z", late_start, runs=[_run("opus-5.5/single", None, [0.25])], news_cost=1.0)
    )
    assert journal.llm_spend_on("2026-10-17") == pytest.approx(0.65)
    assert journal.llm_spend_on("2026-10-18") == pytest.approx(1.25)
    assert journal.llm_spend_on("2026-10-19") == 0.0


def test_unresolved_forecasts_are_mature_and_unresolved(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    made = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    due = _forecast("due", resolves_at=made)
    later = _forecast("later", resolves_at=made + timedelta(hours=4))
    done = _forecast("done", resolves_at=made)
    journal.append_forecasts([due, later, done])
    journal.append_resolutions(
        [Resolution(forecast_id="done", resolved_at=made + timedelta(hours=2), price=1.0, outcome=True)]
    )
    found = journal.unresolved_forecasts(made + timedelta(hours=1))
    assert [row.forecast_id for row in found] == ["due"]


def test_track_record_contents(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    config_id = "opus-5.5/single"
    base = make_snapshot(now=NOW)
    current = _with_marks(base, {"ETH": 210.0, "xyz:GOLD": 4100.0}, "2026-10-17T16:00Z", NOW)
    rows = [
        ("2026-10-16T08:00Z", _decision([("BTC", 0.2, 0.7, "oldest")]), {"BTC": 100.0}),
        (
            "2026-10-16T12:00Z",
            _decision([("ETH", 0.1, 0.62, "second"), ("BTC", 0.0, 0.5, "flat")]),
            {"ETH": 200.0},
        ),
        ("2026-10-16T16:00Z", None, {}),
        ("2026-10-17T12:00Z", _decision([("xyz:GOLD", 0.05, 0.55, "newest")]), {"xyz:GOLD": 4000.0}),
    ]
    for cycle_id, decision, marks in rows:
        moment = _at(cycle_id)
        runs = [_run(config_id, decision, [])]
        if cycle_id == "2026-10-16T12:00Z":
            runs.append(_run("haiku-5.5/single", _decision([("xyz:TSLA", 0.2, 0.7, "other")]), []))
        snapshot = _with_marks(base, marks, cycle_id, moment)
        journal.write_cycle(_cycle(cycle_id, moment, runs=runs, snapshot=snapshot))
    journal.append_forecasts(
        [
            _forecast("good", config_id=config_id, prob_up=1.0),
            _forecast("bad", config_id=config_id, prob_up=1.0),
            _forecast("open", config_id=config_id, prob_up=0.2),
            _forecast("other", config_id="haiku-5.5/single", prob_up=0.0),
        ]
    )
    journal.append_resolutions(
        [
            Resolution(forecast_id="good", resolved_at=NOW, price=110.0, outcome=True),
            Resolution(forecast_id="bad", resolved_at=NOW, price=90.0, outcome=False),
            Resolution(forecast_id="other", resolved_at=NOW, price=110.0, outcome=True),
        ]
    )
    track = journal.track_record(config_id, current)
    assert [(row.cycle_id, row.symbol, row.target_weight, row.thesis) for row in track.recent] == [
        ("2026-10-16T12:00Z", "ETH", 0.1, "second"),
        ("2026-10-17T12:00Z", "xyz:GOLD", 0.05, "newest"),
    ]
    eth, gold = track.recent
    assert eth.prob_up_24h == 0.62
    assert eth.price_then == 200.0
    assert eth.price_now == 210.0
    assert gold.prob_up_24h == 0.55
    assert gold.price_then == 4000.0
    assert gold.price_now == 4100.0
    assert track.resolved_forecasts == 2
    assert track.brier == pytest.approx(0.5)
    quiet = journal.track_record(config_id, current, cycles=0)
    assert quiet.recent == []
    assert quiet.resolved_forecasts == 2
    assert quiet.brier == pytest.approx(0.5)
    missing = journal.track_record("missing", current)
    assert missing.recent == []
    assert missing.resolved_forecasts == 0
    assert missing.brier is None


def test_bookkeeping_rolls_the_utc_day_and_tracks_high_water() -> None:
    opened = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    book = Bookkeeping(
        day="2026-10-16",
        day_start_equity=10000.0,
        high_water_mark=11000.0,
        last_flip_at={"BTC": opened},
    )
    same = update_bookkeeping(book, 10500.0, opened)
    assert same.day == "2026-10-16"
    assert same.day_start_equity == 10000.0
    assert same.high_water_mark == 11000.0
    assert same.last_flip_at == {"BTC": opened}
    higher = update_bookkeeping(book, 12000.0, opened)
    assert higher.day_start_equity == 10000.0
    assert higher.high_water_mark == 12000.0
    nxt = datetime(2026, 10, 17, 0, 30, tzinfo=UTC)
    rolled = update_bookkeeping(book, 9000.0, nxt)
    assert rolled.day == "2026-10-17"
    assert rolled.day_start_equity == 9000.0
    assert rolled.high_water_mark == 11000.0
    assert rolled.last_flip_at == {"BTC": opened}
    lifted = update_bookkeeping(book, 13000.0, nxt)
    assert lifted.day_start_equity == 13000.0
    assert lifted.high_water_mark == 13000.0
    evening = datetime(2026, 10, 16, 22, 0, tzinfo=ZoneInfo("America/New_York"))
    assert update_bookkeeping(book, 10000.0, evening).day == "2026-10-17"
    fresh = update_bookkeeping(None, 8000.0, opened)
    assert fresh == Bookkeeping(day="2026-10-16", day_start_equity=8000.0, high_water_mark=8000.0)


def test_record_flips_only_when_the_sign_changes() -> None:
    now = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
    earlier = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    book = Bookkeeping(
        day="2026-10-17",
        day_start_equity=10000.0,
        high_water_mark=10000.0,
        last_flip_at={"ETH": earlier},
    )
    before = _account({"BTC": 1.0, "ETH": 1.0, "xyz:TSLA": 1.0, "xyz:AAPL": -1.0})
    after = _account({"BTC": -2.0, "ETH": 2.0, "xyz:AAPL": 1.0, "xyz:GOLD": 1.0})
    updated = record_flips(book, before, after, now)
    assert book.last_flip_at == {"ETH": earlier}
    assert updated.last_flip_at == {"ETH": earlier, "BTC": now, "xyz:AAPL": now}
    assert updated.day_start_equity == 10000.0


def test_resolve_waits_for_the_next_hour_and_the_candle() -> None:
    resolves_at = datetime(2026, 10, 17, 12, 30, tzinfo=UTC)
    forecast = _forecast(resolves_at=resolves_at, ref_price=100.0)
    candle = _candle(datetime(2026, 10, 17, 12, 0, tzinfo=UTC), 110.0)
    too_soon = datetime(2026, 10, 17, 13, 29, 59, tzinfo=UTC)
    assert resolve(forecast, [candle], too_soon) is None
    ready = datetime(2026, 10, 17, 13, 30, tzinfo=UTC)
    assert resolve(forecast, [], ready) is None
    wrong_hour = _candle(datetime(2026, 10, 17, 13, 0, tzinfo=UTC), 90.0)
    assert resolve(forecast, [wrong_hour], ready) is None


def test_resolve_outcome_is_the_hourly_close_above_the_reference() -> None:
    resolves_at = datetime(2026, 10, 17, 8, 30, tzinfo=ZoneInfo("America/New_York"))
    forecast = _forecast(resolves_at=resolves_at, ref_price=100.0)
    ready = resolves_at + timedelta(hours=1)
    opened = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
    up = resolve(forecast, [_candle(opened, 100.01), _candle(opened + timedelta(hours=1), 1.0)], ready)
    assert up is not None
    assert up.outcome is True
    assert up.price == 100.01
    assert up.resolved_at == ready
    assert up.forecast_id == forecast.forecast_id
    flat = resolve(forecast, [_candle(opened, 100.0)], ready)
    assert flat is not None
    assert flat.outcome is False
    down = resolve(forecast, [_candle(opened, 99.0)], ready)
    assert down is not None
    assert down.outcome is False


def test_invalid_cycle_id_raises(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    with pytest.raises(ValueError, match="UTC slot"):
        journal.has_cycle("nope")
    with pytest.raises(ValueError, match="UTC slot"):
        journal.has_cycle("2026-13-40T99:00Z")
