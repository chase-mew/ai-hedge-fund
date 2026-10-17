"""Approve a Hyperliquid agent and enable unified account mode."""

from __future__ import annotations

import argparse
import getpass
import json
from typing import Literal

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

Network = Literal["testnet", "mainnet"]


def main() -> None:
    """Read the main wallet key with getpass, approve an agent, and print that key once."""
    parser = argparse.ArgumentParser(description="Approve a Hyperliquid agent and set unified account mode.")
    parser.add_argument("--network", choices=("testnet", "mainnet"), required=True)
    parser.add_argument("--name", default="fund-agent")
    args = parser.parse_args()
    network: Network = args.network
    secret = getpass.getpass("Main wallet private key: ")
    wallet = Account.from_key(secret)
    del secret
    print(f"Derived address: {wallet.address}")
    print(f"Network: {network}")
    answer = input("Type yes to approve an agent for this address: ")
    if answer.strip() != "yes":
        print("Aborted.")
        return
    url = constants.MAINNET_API_URL if network == "mainnet" else constants.TESTNET_API_URL
    exchange = Exchange(wallet, url, account_address=wallet.address, perp_dexs=["", "xyz"])
    result, agent_key = exchange.approve_agent(args.name)
    _dump("approve_agent", result)
    if not (isinstance(result, dict) and result.get("status") == "ok"):
        raise SystemExit("Agent approval failed. The agent key was not printed.")
    abstraction = exchange.user_set_abstraction(wallet.address, "unifiedAccount")
    _dump("user_set_abstraction", abstraction)
    info = Info(url, skip_ws=True, perp_dexs=["", "xyz"])
    _dump("abstraction state", info.query_user_abstraction_state(wallet.address))
    env_name = "HL_AGENT_PRIVATE_KEY_TESTNET" if network == "testnet" else "HL_AGENT_PRIVATE_KEY_MAINNET"
    print(f"Agent private key, shown once. Paste it into .env as {env_name}.")
    print(agent_key)
    print("Agent approvals expire after at most 180 days. Rerun this script to rotate.")


def _dump(label: str, payload: object) -> None:
    print(f"\n== {label} ==")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
