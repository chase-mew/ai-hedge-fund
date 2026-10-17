"""Rounding at the exchange boundary, using live szDecimals from the universe."""

from fund.venues.sizing import ioc_limit_price, round_price, round_size, stop_limit_price


def test_round_size_floors() -> None:
    assert round_size(1.2399, 2) == 1.23
    assert round_size(0.123456789, 5) == 0.12345
    assert round_size(1.2300000001, 2) == 1.23


def test_round_price_jpy_and_btc() -> None:
    assert round_price(157.775, 2) == 157.78
    assert round_price(81491.37, 5) == 81491.0
    assert round_price(7639.44, 3) == 7639.4


def test_ioc_and_stop_limits() -> None:
    assert ioc_limit_price(100.0, "buy", 0.01, 3) == round_price(101.0, 3)
    assert ioc_limit_price(100.0, "sell", 0.01, 3) == round_price(99.0, 3)
    assert stop_limit_price(100.0, "sell", 2) == round_price(90.0, 2)
    assert stop_limit_price(100.0, "buy", 2) == round_price(110.0, 2)
