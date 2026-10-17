"""Paper accounting: fills, funding, stops, and equity conservation."""

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from fund.config import PaperSettings
from fund.contracts import Candle, MarketMeta, MarketQuote, PaperAccount, PaperPosition, Side, StopOrder
from fund.venues.paper import (
    PaperVenue,
    accrue_funding,
    new_paper_account,
    paper_equity,
    process_stops,
)

NOW = datetime(2026, 10, 14, 16, 0, tzinfo=UTC)
PAPER = PaperSettings(slippage_bps=5, core_taker_fee=0.00045, xyz_taker_fee=0.00009)
META = {"BTC": MarketMeta(symbol="BTC", sz_decimals=5, max_leverage=40, only_isolated=False)}


def test_new_account_starts_flat() -> None:
    account = new_paper_account("config:opus-5.5/single", 10_000, NOW)
    assert account.cash == 10_000
    assert account.positions == {}
    assert account.fees_cum == 0
    assert account.funding_cum == 0
    assert account.updated_at == NOW
    assert account.bookkeeping.day == "2026-10-14"
    assert account.bookkeeping.day_start_equity == 10_000
    assert account.bookkeeping.high_water_mark == 10_000
    venue = PaperVenue(account, {}, {}, PAPER, NOW)
    assert venue.name == "paper"
    assert venue.marks() == {}
    assert venue.market_meta() == {}
    snapshot = venue.account()
    assert snapshot.equity == 10_000
    assert snapshot.positions == {}
    assert venue.state() == account


def test_open_add_reduce_flip_and_close_match_hand_numbers() -> None:
    venue = PaperVenue(new_paper_account("p", 10_000, NOW), {"BTC": 80_000}, META, PAPER, NOW)
    opened = venue.market_order("BTC", "buy", 0.1, False)
    assert opened.status == "filled"
    assert opened.order_id == "paper-1"
    assert opened.avg_px == pytest.approx(80_040)
    assert opened.fee_usd == pytest.approx(3.6018)
    assert venue.state().cash == pytest.approx(9_996.3982)
    assert venue.state().fees_cum == pytest.approx(3.6018)
    assert venue.state().positions["BTC"].entry_px == pytest.approx(80_040)
    assert venue.account().equity == pytest.approx(9_992.3982)
    assert venue.account().positions["BTC"].mark == 80_000

    added = venue.market_order("BTC", "buy", 0.05, False)
    assert added.status == "filled"
    assert added.order_id == "paper-2"
    assert added.fee_usd == pytest.approx(1.8009)
    assert venue.state().cash == pytest.approx(9_994.5973)
    assert venue.state().fees_cum == pytest.approx(5.4027)
    assert venue.state().positions["BTC"].size == pytest.approx(0.15)
    assert venue.state().positions["BTC"].entry_px == pytest.approx(80_040)
    assert venue.account().equity == pytest.approx(9_988.5973)

    marked = PaperVenue(venue.state(), {"BTC": 81_000}, META, PAPER, NOW + timedelta(hours=1))
    assert marked.account().equity == pytest.approx(10_138.5973)

    reduced = marked.market_order("BTC", "sell", 0.05, False)
    assert reduced.status == "filled"
    assert reduced.avg_px == pytest.approx(80_959.5)
    assert reduced.fee_usd == pytest.approx(1.82158875)
    assert marked.state().cash == pytest.approx(10_038.75071125)
    assert marked.state().positions["BTC"].size == pytest.approx(0.1)
    assert marked.state().positions["BTC"].entry_px == pytest.approx(80_040)
    assert marked.account().equity == pytest.approx(10_134.75071125)

    flipped = marked.market_order("BTC", "sell", 0.25, False)
    assert flipped.status == "filled"
    assert flipped.filled_size == pytest.approx(0.25)
    assert marked.state().positions["BTC"].size == pytest.approx(-0.15)
    assert marked.state().positions["BTC"].entry_px == pytest.approx(80_959.5)
    assert marked.state().positions["BTC"].stop is None
    assert marked.state().cash == pytest.approx(10_121.5927675)
    assert marked.account().equity == pytest.approx(10_115.5177675)

    closed = marked.close_position("BTC")
    assert closed.status == "filled"
    assert closed.side == "buy"
    assert closed.filled_size == pytest.approx(0.15)
    assert closed.avg_px == pytest.approx(81_040.5)
    assert "BTC" not in marked.state().positions
    assert marked.state().cash == pytest.approx(10_103.97253375)
    assert marked.state().fees_cum == pytest.approx(21.80246625)
    assert marked.account().equity == pytest.approx(10_103.97253375)
    assert marked.state().bookkeeping == venue.state().bookkeeping
    assert marked.state().last_marks["BTC"] == 81_000


def test_xyz_symbol_uses_the_xyz_taker_fee() -> None:
    meta = {"xyz:GOLD": MarketMeta(symbol="xyz:GOLD", sz_decimals=4, max_leverage=25, only_isolated=False)}
    venue = PaperVenue(new_paper_account("p", 10_000, NOW), {"xyz:GOLD": 4_000}, meta, PAPER, NOW)
    outcome = venue.market_order("xyz:GOLD", "buy", 1, False)
    assert outcome.avg_px == pytest.approx(4_002)
    assert outcome.fee_usd == pytest.approx(0.36018)
    assert venue.state().cash == pytest.approx(9_999.63982)
    assert venue.state().fees_cum == pytest.approx(0.36018)


def test_short_add_reduce_and_flip_match_hand_numbers() -> None:
    venue = PaperVenue(new_paper_account("p", 10_000, NOW), {"BTC": 100}, META, PAPER, NOW)
    venue.market_order("BTC", "sell", 2, False)
    assert venue.state().positions["BTC"].size == pytest.approx(-2)
    assert venue.state().positions["BTC"].entry_px == pytest.approx(99.95)
    assert venue.state().cash == pytest.approx(9_999.910045)

    remark = PaperVenue(venue.state(), {"BTC": 90}, META, PAPER, NOW)
    remark.market_order("BTC", "sell", 2, False)
    assert remark.state().positions["BTC"].size == pytest.approx(-4)
    assert remark.state().positions["BTC"].entry_px == pytest.approx(94.9525)
    assert remark.state().cash == pytest.approx(9_999.8290855)

    reduced = remark.market_order("BTC", "buy", 1, False)
    assert reduced.side == "buy"
    assert remark.state().positions["BTC"].size == pytest.approx(-3)
    assert remark.state().positions["BTC"].entry_px == pytest.approx(94.9525)
    assert remark.state().cash == pytest.approx(10_004.69606525)

    remark.market_order("BTC", "buy", 5, False)
    assert remark.state().positions["BTC"].size == pytest.approx(2)
    assert remark.state().positions["BTC"].entry_px == pytest.approx(90.045)
    assert remark.state().cash == pytest.approx(10_019.215964)
    assert remark.account().equity == pytest.approx(10_019.125964)


def test_reduce_only_cannot_increase_or_cross_zero() -> None:
    venue = PaperVenue(new_paper_account("p", 10_000, NOW), {"BTC": 100}, META, PAPER, NOW)
    venue.market_order("BTC", "buy", 1, False)
    cash = venue.state().cash
    blocked = venue.market_order("BTC", "buy", 1, True)
    assert blocked.status == "unfilled"
    assert blocked.filled_size == 0
    assert blocked.avg_px is None
    assert venue.state().cash == cash
    assert venue.state().positions["BTC"].size == pytest.approx(1)

    clipped = venue.market_order("BTC", "sell", 2, True)
    assert clipped.status == "partial"
    assert clipped.filled_size == pytest.approx(1)
    assert clipped.requested_size == pytest.approx(2)
    assert "BTC" not in venue.state().positions

    flat = venue.market_order("BTC", "sell", 1, True)
    assert flat.status == "unfilled"


def test_close_position_sells_a_long_and_skips_a_flat_book() -> None:
    venue = PaperVenue(new_paper_account("p", 10_000, NOW), {"BTC": 100}, META, PAPER, NOW)
    skipped = venue.close_position("BTC")
    assert skipped.status == "skipped"
    assert skipped.message == "flat"
    assert skipped.filled_size == 0

    venue.market_order("BTC", "buy", 1.5, False)
    closed = venue.close_position("BTC")
    assert closed.status == "filled"
    assert closed.side == "sell"
    assert closed.filled_size == pytest.approx(1.5)
    assert "BTC" not in venue.state().positions


def test_unfilled_when_size_or_mark_is_unusable() -> None:
    venue = PaperVenue(new_paper_account("p", 10_000, NOW), {"BTC": 0}, META, PAPER, NOW)
    assert venue.market_order("BTC", "buy", 0, False).status == "unfilled"
    assert venue.market_order("ETH", "buy", 1, False).status == "unfilled"
    assert venue.market_order("BTC", "buy", 1, False).status == "unfilled"
    negative = PaperVenue(new_paper_account("p", 10_000, NOW), {"BTC": -1}, META, PAPER, NOW)
    assert negative.market_order("BTC", "sell", 1, False).status == "unfilled"


def test_stop_survives_a_reduce_and_clears_when_the_position_flips() -> None:
    venue = PaperVenue(new_paper_account("p", 10_000, NOW), {"BTC": 100}, META, PAPER, NOW)
    venue.market_order("BTC", "buy", 2, False)
    stop = StopOrder(symbol="BTC", side="sell", size=2, trigger_px=90)
    venue.sync_stop("BTC", stop)
    venue.sync_stop("ETH", stop)
    venue.market_order("BTC", "sell", 0.5, True)
    assert venue.state().positions["BTC"].stop == stop
    venue.market_order("BTC", "sell", 3, False)
    assert venue.state().positions["BTC"].size < 0
    assert venue.state().positions["BTC"].stop is None
    venue.sync_stop("BTC", None)
    assert venue.state().positions["BTC"].stop is None


def test_ensure_leverage_does_not_change_the_account() -> None:
    venue = PaperVenue(new_paper_account("p", 10_000, NOW), {"BTC": 100}, META, PAPER, NOW)
    before = venue.state()
    venue.ensure_leverage("BTC", 5, True)
    assert venue.state() == before


def test_funding_long_pays_and_short_receives() -> None:
    start = new_paper_account("p", 10_000, NOW)
    account = start.model_copy(
        update={
            "positions": {
                "BTC": PaperPosition(symbol="BTC", size=0.5, entry_px=80_000),
                "ETH": PaperPosition(symbol="ETH", size=-2, entry_px=2_400),
                "xyz:GOLD": PaperPosition(symbol="xyz:GOLD", size=1, entry_px=4_000),
            },
            "last_marks": {"BTC": 80_000, "ETH": 2_400},
        }
    )
    quotes = {
        "BTC": _quote("BTC", 81_000, 0.0000125),
        "ETH": _quote("ETH", 2_500, 0.0001),
        "SOL": _quote("SOL", 100, 0.01),
    }
    funded = accrue_funding(account, quotes, NOW + timedelta(hours=4))
    assert funded.cash == pytest.approx(9_999.975)
    assert funded.funding_cum == pytest.approx(0.025)
    assert funded.updated_at == NOW + timedelta(hours=4)
    assert funded.last_marks["BTC"] == 81_000
    assert funded.last_marks["ETH"] == 2_500
    assert funded.positions["xyz:GOLD"].size == pytest.approx(1)
    assert accrue_funding(funded, quotes, funded.updated_at) is funded


def test_funding_ignores_non_positive_elapsed_time() -> None:
    account = new_paper_account("p", 10_000, NOW)
    quotes = {"BTC": _quote("BTC", 80_000, 0.01)}
    assert accrue_funding(account, quotes, NOW) is account
    assert accrue_funding(account, quotes, NOW - timedelta(hours=2)) is account


def test_long_stop_fills_at_the_trigger_minus_slippage() -> None:
    account = _held(
        "BTC",
        0.2,
        80_000,
        StopOrder(symbol="BTC", side="sell", size=0.2, trigger_px=76_000),
    )
    candles = {
        "BTC": [
            _candle(NOW - timedelta(hours=1), low=1, high=90_000),
            _candle(NOW + timedelta(minutes=30), low=77_000, high=81_000),
            _candle(NOW + timedelta(hours=1), low=76_000, high=80_000),
        ]
    }
    done = process_stops(account, candles, PAPER, NOW + timedelta(hours=2))
    assert "BTC" not in done.positions
    assert done.cash == pytest.approx(9_185.56342)
    assert done.fees_cum == pytest.approx(6.83658)
    assert paper_equity(done, {"BTC": 76_000}) == pytest.approx(9_185.56342)


def test_short_stop_fills_at_the_trigger_plus_slippage() -> None:
    account = _held(
        "ETH",
        -1.5,
        2_400,
        StopOrder(symbol="ETH", side="buy", size=1.5, trigger_px=2_500),
        cash=5_000,
    )
    candles = {
        "ETH": [
            _candle(NOW + timedelta(minutes=30), low=2_000, high=2_400),
            _candle(NOW + timedelta(hours=1), low=2_000, high=2_500),
        ]
    }
    done = process_stops(account, candles, PAPER, NOW + timedelta(hours=2))
    assert "ETH" not in done.positions
    assert done.cash == pytest.approx(4_846.43665625)
    assert done.fees_cum == pytest.approx(1.68834375)


def test_stop_ignores_candles_outside_the_window_and_positions_without_stops() -> None:
    stopped = _held("BTC", 1, 100, StopOrder(symbol="BTC", side="sell", size=1, trigger_px=90))
    bare = PaperPosition(symbol="ETH", size=1, entry_px=100)
    account = stopped.model_copy(update={"positions": {**stopped.positions, "ETH": bare}})
    candles = {
        "BTC": [
            _candle(NOW, low=1, high=200),
            _candle(NOW + timedelta(hours=3), low=1, high=200),
            _candle(NOW + timedelta(hours=1), low=95, high=105),
        ],
        "ETH": [_candle(NOW + timedelta(hours=1), low=1, high=200)],
    }
    done = process_stops(account, candles, PAPER, NOW + timedelta(hours=2))
    assert done.cash == account.cash
    assert done.positions["BTC"].size == pytest.approx(1)
    assert done.positions["ETH"].size == pytest.approx(1)
    assert process_stops(account, {}, PAPER, NOW + timedelta(hours=2)).positions["BTC"].stop is not None


def test_equity_falls_back_to_last_mark_then_entry() -> None:
    account = new_paper_account("p", 1_000, NOW).model_copy(
        update={
            "positions": {"BTC": PaperPosition(symbol="BTC", size=1, entry_px=100)},
            "last_marks": {"BTC": 110},
        }
    )
    assert paper_equity(account, {}) == pytest.approx(1_010)
    assert paper_equity(account, {"BTC": 120}) == pytest.approx(1_020)
    unmarked = account.model_copy(update={"last_marks": {}})
    assert paper_equity(unmarked, {}) == pytest.approx(1_000)


@given(
    slip_bps=st.floats(min_value=0, max_value=25, allow_nan=False, allow_infinity=False),
    fee_rate=st.floats(min_value=0, max_value=0.002, allow_nan=False, allow_infinity=False),
    steps=st.lists(
        st.one_of(
            st.tuples(
                st.just("mark"),
                st.floats(min_value=20, max_value=5_000, allow_nan=False, allow_infinity=False),
            ),
            st.tuples(
                st.just("buy"),
                st.floats(min_value=0.05, max_value=2, allow_nan=False, allow_infinity=False),
            ),
            st.tuples(
                st.just("sell"),
                st.floats(min_value=0.05, max_value=2, allow_nan=False, allow_infinity=False),
            ),
        ),
        min_size=1,
        max_size=8,
    ),
)
@hypothesis_settings(max_examples=80, deadline=None)
def test_equity_conserved_apart_from_fees_and_price_moves(
    slip_bps: float,
    fee_rate: float,
    steps: list[tuple[str, float]],
) -> None:
    settings = PaperSettings(slippage_bps=slip_bps, core_taker_fee=fee_rate, xyz_taker_fee=fee_rate)
    mark = 100.0
    venue = PaperVenue(new_paper_account("baseline:cash", 1_000_000, NOW), {"BTC": mark}, META, settings, NOW)
    slip = slip_bps / 10_000
    fees = 0.0
    for kind, value in steps:
        if kind == "mark":
            position = venue.state().positions.get("BTC")
            size = 0.0 if position is None else position.size
            expected = venue.account().equity + size * (value - mark)
            mark = value
            venue = PaperVenue(venue.state(), {"BTC": mark}, META, settings, NOW)
            assert venue.account().equity == pytest.approx(expected, rel=1e-8, abs=1e-4)
            continue
        side: Side = "buy" if kind == "buy" else "sell"
        before = venue.account().equity
        outcome = venue.market_order("BTC", side, value, False)
        fees += outcome.fee_usd
        expected = before - outcome.filled_size * mark * slip - outcome.fee_usd
        assert venue.account().equity == pytest.approx(expected, rel=1e-8, abs=1e-4)
        assert venue.state().fees_cum == pytest.approx(fees, rel=1e-8, abs=1e-4)


def _quote(symbol: str, mark: float, funding: float) -> MarketQuote:
    return MarketQuote(
        symbol=symbol,
        mark=mark,
        oracle=mark,
        prev_day_px=mark,
        funding_hourly=funding,
        open_interest=0,
        day_volume_usd=0,
    )


def _candle(open_time: datetime, low: float, high: float) -> Candle:
    return Candle(open_time=open_time, open=low, high=high, low=low, close=high, volume=0)


def _held(symbol: str, size: float, entry: float, stop: StopOrder, cash: float = 10_000) -> PaperAccount:
    account = new_paper_account("p", cash, NOW)
    position = PaperPosition(symbol=symbol, size=size, entry_px=entry, stop=stop)
    return account.model_copy(update={"positions": {symbol: position}})
