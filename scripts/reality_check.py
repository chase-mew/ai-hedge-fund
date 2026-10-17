"""Phase 0 reality check. Reads public market data and, if a key is set, the Claude API."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from dotenv import load_dotenv
from hyperliquid.info import Info
from hyperliquid.utils import constants
from pydantic import BaseModel, Field

UNIVERSE = [
    "xyz:SP500",
    "xyz:XYZ100",
    "xyz:NVDA",
    "xyz:AAPL",
    "xyz:MSFT",
    "xyz:TSLA",
    "xyz:GOLD",
    "xyz:CL",
    "xyz:JPY",
    "BTC",
    "ETH",
]

MODELS = [
    ("opus-5.5", "claude-opus-5-5"),
    ("sonnet-5.5", "claude-sonnet-5-5"),
    ("haiku-5.5", "claude-haiku-5-5"),
]


class TinyDecision(BaseModel):
    note: str = Field(description="One short sentence.")
    confidence: float = Field(ge=0.0, le=1.0, description="A number between 0 and 1.")


def info_client(url: str) -> Info:
    return Info(url, skip_ws=True, perp_dexs=["", "xyz"])


def fetch_meta(info: Info) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for dex in ("", "xyz"):
        payload = info.post("/info", {"type": "metaAndAssetCtxs", "dex": dex})
        meta, ctxs = payload
        for asset, ctx in zip(meta["universe"], ctxs, strict=True):
            rows[asset["name"]] = {
                "szDecimals": asset["szDecimals"],
                "maxLeverage": asset["maxLeverage"],
                "onlyIsolated": bool(asset.get("onlyIsolated", False)),
                "mark": ctx["markPx"],
                "oracle": ctx["oraclePx"],
                "prevDayPx": ctx["prevDayPx"],
                "funding": ctx["funding"],
                "dayNtlVlm": ctx["dayNtlVlm"],
            }
    return rows


def print_table(title: str, rows: dict[str, dict[str, object]]) -> None:
    print(f"\n== {title} ==")
    header = (
        f"{'symbol':<12} {'sz':>3} {'lev':>4} {'iso':>5} {'mark':>12} {'oracle':>12} "
        f"{'prev':>12} {'funding':>12} {'dayNtl':>14}"
    )
    print(header)
    missing = []
    for symbol in UNIVERSE:
        row = rows.get(symbol)
        if row is None:
            missing.append(symbol)
            print(f"{symbol:<12} MISSING")
            continue
        print(
            f"{symbol:<12} {row['szDecimals']:>3} {row['maxLeverage']:>4} "
            f"{row['onlyIsolated']!s:>5} {row['mark']:>12} {row['oracle']:>12} "
            f"{row['prevDayPx']:>12} {row['funding']:>12} {row['dayNtlVlm']:>14}"
        )
    if missing:
        raise SystemExit(f"Missing symbols on {title}: {', '.join(missing)}")


def check_candles(info: Info) -> None:
    end = datetime.now(UTC)
    start = end - timedelta(days=120)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    for symbol in ("xyz:SP500", "BTC"):
        candles = info.candles_snapshot(symbol, "1d", start_ms, end_ms)
        times = [int(c["t"]) for c in candles]
        ordered = times == sorted(times)
        first = candles[0] if candles else None
        last = candles[-1] if candles else None
        print(f"\n== candles {symbol} count={len(candles)} ascending={ordered} ==")
        print(f"first: {first}")
        print(f"last: {last}")
        if not ordered:
            raise SystemExit(f"Candles for {symbol} are not in ascending order")
        if len(candles) < 60:
            raise SystemExit(f"Too few candles for {symbol}: {len(candles)}")


def print_differences(mainnet: dict[str, dict[str, object]], testnet: dict[str, dict[str, object]]) -> None:
    print("\n== mainnet vs testnet metadata ==")
    for symbol in UNIVERSE:
        a = mainnet[symbol]
        b = testnet.get(symbol)
        if b is None:
            print(f"{symbol}: missing on testnet")
            continue
        diffs = []
        for key in ("szDecimals", "maxLeverage", "onlyIsolated"):
            if a[key] != b[key]:
                diffs.append(f"{key} mainnet={a[key]} testnet={b[key]}")
        if diffs:
            print(f"{symbol}: {'; '.join(diffs)}")
        else:
            print(f"{symbol}: metadata matches")


def thinking_blocks(response: object) -> list[str]:
    texts: list[str] = []
    content = getattr(response, "content", []) or []
    for block in content:
        if getattr(block, "type", None) == "thinking":
            thinking = getattr(block, "thinking", None) or getattr(block, "text", None)
            if thinking:
                texts.append(str(thinking))
    return texts


def try_parse(client: Any, model_id: str, thinking: dict[str, str] | None, effort: bool) -> Any:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": 1024,
        "system": "Reply with the structured object only.",
        "messages": [{"role": "user", "content": "note: ok. confidence: 0.5"}],
        "output_format": TinyDecision,
    }
    if effort:
        kwargs["output_config"] = {"effort": "medium"}
    if thinking is not None:
        kwargs["thinking"] = thinking
    return client.messages.parse(**kwargs)


def check_models(api_key: str) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=300, max_retries=1)
    for key, model_id in MODELS:
        print(f"\n== model {key} {model_id} ==")
        attempts = [
            ("thinking+effort", {"type": "adaptive", "display": "summarized"}, True),
            ("effort only", None, True),
            ("bare", None, False),
        ]
        last_error: Exception | None = None
        for label, thinking, effort in attempts:
            try:
                response = try_parse(client, model_id, thinking, effort)
            except anthropic.APIStatusError as exc:
                last_error = exc
                print(f"  {label}: HTTP {exc.status_code} {exc.message}")
                if exc.status_code != 400:
                    break
                continue
            parsed = response.parsed_output
            summaries = thinking_blocks(response)
            print(f"  combo: {label}")
            print(f"  stop_reason: {response.stop_reason}")
            print(f"  usage: {response.usage}")
            print(f"  parsed_ok: {isinstance(parsed, TinyDecision)} value={parsed}")
            print(f"  thinking_summary_nonempty: {any(s.strip() for s in summaries)}")
            last_error = None
            break
        if last_error is not None:
            print(f"  FAILED: {last_error}")

    print("\n== web search ==")
    search_user = "Give one factual headline about the gold price today. One sentence."
    for tool_type in ("web_search_20260318", "web_search_20250305"):
        try:
            tools: Any = [{"type": tool_type, "name": "web_search", "max_uses": 2}]
            response = client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=1024,
                tools=tools,
                messages=[{"role": "user", "content": search_user}],
            )
        except anthropic.APIStatusError as exc:
            print(f"  {tool_type}: HTTP {exc.status_code} {exc.message}")
            continue
        searches = 0
        server_use = getattr(response.usage, "server_tool_use", None)
        if server_use is not None:
            searches = getattr(server_use, "web_search_requests", 0) or 0
        citation = None
        for block in response.content:
            citations = getattr(block, "citations", None) or []
            for item in citations:
                url = getattr(item, "url", None)
                if url:
                    citation = url
                    break
            if citation:
                break
        print(f"  tool: {tool_type}")
        print(f"  stop_reason: {response.stop_reason}")
        print(f"  web_search_requests: {searches}")
        print(f"  first_citation: {citation}")
        break

    counted = client.messages.count_tokens(
        model="claude-sonnet-5-5",
        messages=[{"role": "user", "content": "Count these tokens."}],
    )
    print(f"\n== count_tokens ==\n  {counted}")


def main() -> None:
    load_dotenv()
    print(f"now_utc: {datetime.now(UTC).isoformat()}")
    mainnet = info_client(constants.MAINNET_API_URL)
    main_rows = fetch_meta(mainnet)
    print_table("mainnet", main_rows)
    check_candles(mainnet)
    testnet = info_client(constants.TESTNET_API_URL)
    test_rows = fetch_meta(testnet)
    print_table("testnet", test_rows)
    print_differences(main_rows, test_rows)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("\n== claude ==\nANTHROPIC_API_KEY is not set. Market checks finished; model checks skipped.")
        raise SystemExit(2)
    check_models(api_key)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
