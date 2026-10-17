# pyright: strict
"""Save live Hyperliquid mainnet responses as test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fund.config import load_config
from fund.data.hyperliquid_data import HyperliquidData, recorded_client
from fund.data.snapshot import build_snapshot


def main() -> None:
    """Fetch mainnet meta, candles, and a snapshot into tests/fixtures."""
    root = Path(__file__).resolve().parents[1]
    out = root / "tests" / "fixtures"
    config = load_config(root / "fund.toml")
    symbols = [spec.symbol for spec in config.universe]
    now = datetime.now(UTC)
    HyperliquidData().write_raw(out, symbols, now)
    snapshot = build_snapshot(recorded_client(out), list(config.universe), now, config.fund.cycle_hours)
    (out / "snapshot_real.json").write_text(snapshot.model_dump_json(indent=2) + "\n")
    print(f"Wrote Hyperliquid fixtures to {out}")


if __name__ == "__main__":
    main()
