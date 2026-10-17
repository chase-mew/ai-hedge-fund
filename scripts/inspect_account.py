"""Print raw Hyperliquid account JSON and the computed account snapshot."""

from __future__ import annotations

import argparse
import json
from typing import Literal

from fund.config import load_config, load_secrets
from fund.venues.hyperliquid import HyperliquidVenue

Network = Literal["testnet", "mainnet"]


def main() -> None:
    """Print abstraction, both dex states, spot, open orders, and account()."""
    parser = argparse.ArgumentParser(description="Print raw account JSON and the computed account snapshot.")
    parser.add_argument("--network", choices=("testnet", "mainnet"), default="testnet")
    args = parser.parse_args()
    network: Network = args.network
    address, key, allow_mainnet = _credentials(network)
    config = load_config()
    symbols = [item.symbol for item in config.universe]
    if network == "testnet":
        slippage = config.execution.testnet_slippage
    else:
        slippage = config.execution.mainnet_slippage
    venue = HyperliquidVenue(network, address, key, symbols, slippage, allow_mainnet)
    info = venue.info
    _dump("query_user_abstraction_state", info.query_user_abstraction_state(address))
    for dex in ("", "xyz"):
        label = "user_state core" if dex == "" else "user_state xyz"
        _dump(label, info.user_state(address, dex))
    _dump("spot_user_state", info.spot_user_state(address))
    for dex in ("", "xyz"):
        label = "open_orders core" if dex == "" else "open_orders xyz"
        _dump(label, info.frontend_open_orders(address, dex))
    print("Computed account() uses the larger marginSummary.accountValue and does not add both dex values.")
    _dump("account", venue.account().model_dump(mode="json"))


def _credentials(network: Network) -> tuple[str, str, bool]:
    secrets = load_secrets()
    address = secrets.hl_account_address
    if not address:
        raise SystemExit("HL_ACCOUNT_ADDRESS is not set. Add the main wallet address to .env.")
    if network == "testnet":
        key = secrets.hl_agent_private_key_testnet
        env_name = "HL_AGENT_PRIVATE_KEY_TESTNET"
    else:
        key = secrets.hl_agent_private_key_mainnet
        env_name = "HL_AGENT_PRIVATE_KEY_MAINNET"
    if not key:
        raise SystemExit(
            f"{env_name} is not set. Run scripts/setup_agent.py --network {network} and add the key to .env."
        )
    return address, key, secrets.allow_mainnet


def _dump(label: str, payload: object) -> None:
    print(f"\n== {label} ==")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
