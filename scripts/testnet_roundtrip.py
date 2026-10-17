"""Buy and flatten BTC and the most liquid xyz market on testnet."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any, cast

from fund.config import load_config, load_secrets
from fund.contracts import MarketMeta, OrderOutcome, StopOrder
from fund.venues.hyperliquid import HyperliquidVenue
from fund.venues.sizing import round_price, round_size


def main() -> None:
    """Run a small testnet round trip. This script refuses mainnet."""
    parser = argparse.ArgumentParser(description="Place a small testnet round trip. Refuses mainnet.")
    parser.add_argument("--network", choices=("testnet", "mainnet"), default="testnet")
    args = parser.parse_args()
    if args.network != "testnet":
        raise SystemExit("This script refuses mainnet. Use --network testnet.")
    address, key = _credentials()
    config = load_config()
    symbols = [item.symbol for item in config.universe]
    venue = HyperliquidVenue(
        "testnet",
        address,
        key,
        symbols,
        config.execution.testnet_slippage,
        False,
    )
    started = datetime.now(UTC)
    equity_before = venue.account().equity
    print(f"equity {equity_before:.2f}")
    xyz_symbol = _most_liquid_xyz(venue)
    for symbol in ("BTC", xyz_symbol):
        meta = venue.market_meta()[symbol]
        _leg(venue, symbol, meta, config.execution.exchange_leverage, xyz=symbol.startswith("xyz:"))
    equity_after = venue.account().equity
    print(f"fees {json.dumps(venue.fees_since(started))}")
    print(f"equity change {equity_after - equity_before:.4f}")


def _leg(
    venue: HyperliquidVenue,
    symbol: str,
    meta: MarketMeta,
    exchange_leverage: int,
    *,
    xyz: bool,
) -> None:
    print(f"equity {venue.account().equity:.2f}")
    print(f"leg {symbol}")
    leverage = min(exchange_leverage, meta.max_leverage)
    venue.ensure_leverage(symbol, leverage, cross=not meta.only_isolated)
    mid = _mid(venue, symbol)
    size = round_size(15.0 / mid, meta.sz_decimals)
    if size * mid < 11:
        size = round_size(11.0 / mid, meta.sz_decimals)
    if size <= 0:
        raise SystemExit(f"{symbol} size rounds to zero at mid {mid}.")
    outcome = venue.market_order(symbol, "buy", size, False)
    _print_outcome("buy", outcome)
    if _unfilled(outcome, xyz=xyz, action="buy"):
        return
    position = venue.account().positions.get(symbol)
    if position is None or position.size <= 0:
        if xyz:
            print(f"{symbol} shows no position after the order. Reported, not a failure.")
            return
        raise SystemExit(f"{symbol} position was not opened.")
    trigger = round_price(position.entry_px * 0.95, meta.sz_decimals)
    venue.sync_stop(
        symbol,
        StopOrder(symbol=symbol, side="sell", size=abs(position.size), trigger_px=trigger),
    )
    if not _triggers(venue, symbol):
        raise SystemExit(f"{symbol} stop was not listed.")
    closed = venue.close_position(symbol)
    _print_outcome("close", closed)
    if _unfilled(closed, xyz=xyz, action="close"):
        return
    venue.sync_stop(symbol, None)
    remaining = venue.account().positions.get(symbol)
    if remaining is not None and remaining.size != 0:
        if xyz:
            print(f"{symbol} is not flat after the close. Reported, not a failure.")
            return
        raise SystemExit(f"{symbol} is not flat.")
    if _triggers(venue, symbol):
        raise SystemExit(f"{symbol} still has a trigger order.")
    print(f"{symbol} flat")


def _unfilled(outcome: OrderOutcome, *, xyz: bool, action: str) -> bool:
    if outcome.status in ("filled", "partial"):
        return False
    if outcome.status == "unfilled" and xyz:
        print(
            f"{outcome.symbol} {action} did not fill. "
            "Thin xyz books often have no liquidity. Reported, not a failure."
        )
        return True
    raise SystemExit(f"{outcome.symbol} {action} failed ({outcome.status}): {outcome.message}")


def _most_liquid_xyz(venue: HyperliquidVenue) -> str:
    payload = cast(list[Any], venue.info.post("/info", {"type": "metaAndAssetCtxs", "dex": "xyz"}))
    meta = cast(dict[str, Any], payload[0])
    ctxs = cast(list[dict[str, Any]], payload[1])
    best_name = ""
    best_volume = -1.0
    for asset, ctx in zip(meta["universe"], ctxs, strict=True):
        volume = float(ctx["dayNtlVlm"])
        if volume > best_volume:
            best_volume = volume
            best_name = str(asset["name"])
    if not best_name:
        raise SystemExit("No xyz markets were returned.")
    print(f"most liquid xyz market {best_name} day volume {best_volume:.2f}")
    return best_name


def _mid(venue: HyperliquidVenue, symbol: str) -> float:
    dex = symbol.split(":", 1)[0] if ":" in symbol else ""
    mids = cast(dict[str, Any], venue.info.all_mids(dex))
    return float(mids[symbol])


def _triggers(venue: HyperliquidVenue, symbol: str) -> list[dict[str, Any]]:
    dex = symbol.split(":", 1)[0] if ":" in symbol else ""
    orders = cast(list[dict[str, Any]], venue.info.frontend_open_orders(venue.account_address, dex))
    return [
        order
        for order in orders
        if order.get("coin") == symbol and order.get("isTrigger") and order.get("reduceOnly")
    ]


def _print_outcome(label: str, outcome: OrderOutcome) -> None:
    print(
        f"{label} {outcome.symbol} {outcome.status} filled {outcome.filled_size} "
        f"avg {outcome.avg_px} id {outcome.order_id} {outcome.message}"
    )


def _credentials() -> tuple[str, str]:
    secrets = load_secrets()
    address = secrets.hl_account_address
    if not address:
        raise SystemExit("HL_ACCOUNT_ADDRESS is not set. Add the main wallet address to .env.")
    key = secrets.hl_agent_private_key_testnet
    if not key:
        raise SystemExit(
            "HL_AGENT_PRIVATE_KEY_TESTNET is not set. "
            "Run scripts/setup_agent.py --network testnet and add the key to .env."
        )
    return address, key


if __name__ == "__main__":
    main()
