"""Command line for one cycle, status, scores, the dashboard, and safety controls."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from fund.brain.decide import decide
from fund.brain.fake import FakeClaude
from fund.brain.llm import Claude
from fund.config import AgentConfig, FundConfig, Secrets, load_config, load_secrets
from fund.contracts import (
    AccountSnapshot,
    Bookkeeping,
    CycleRecord,
    GateResult,
    Mode,
    OrderPlan,
    PortfolioState,
    Snapshot,
)
from fund.cycle import CycleContext, cycle_failed, run_cycle
from fund.dashboard.build import build_site
from fund.data.hyperliquid_data import HyperliquidData
from fund.journal.store import Journal
from fund.risk.gate import apply_gate
from fund.scoring.report import build_report
from fund.venues.hyperliquid import HyperliquidVenue

app = typer.Typer(help="An autonomous multi asset AI fund. Claude proposes, code disposes.")
console = Console()
logger = logging.getLogger(__name__)


@app.callback()
def main() -> None:
    """An autonomous multi asset AI fund. Claude proposes, code disposes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_level=False)],
    )


@app.command()
def run(
    mode: Annotated[str | None, typer.Option("--mode", help="paper, testnet, or live.")] = None,
    fake_llm: Annotated[bool, typer.Option("--fake-llm", help="Use the offline model.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Run again if this cycle is stored.")] = False,
    config_path: Annotated[Path, typer.Option("--config", help="Path to fund.toml.")] = Path("fund.toml"),
) -> None:
    """Run one cycle and print the champion result."""
    config = load_config(config_path)
    secrets = load_secrets()
    chosen = _mode(mode)
    now = datetime.now(UTC)
    journal = _journal(config)
    claude: Claude | FakeClaude = FakeClaude() if fake_llm else Claude()
    venue = None
    approve = None
    if chosen != "paper" and config.fund.autonomy != "paper":
        venue = _venue(config, secrets, chosen)
        if config.fund.autonomy == "approve":
            approve = _confirm
    record = run_cycle(
        CycleContext(
            config=config,
            secrets=secrets,
            journal=journal,
            data=HyperliquidData(),
            claude=claude,
            mode=chosen,
            now=now,
            live_venue=venue,
            approve=approve,
            force=force,
        )
    )
    _print_run(record)
    if cycle_failed(record):
        for item in record.errors:
            console.print(f"[red]{item}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    mode: Annotated[str | None, typer.Option("--mode", help="paper, testnet, or live.")] = None,
    config_path: Annotated[Path, typer.Option("--config", help="Path to fund.toml.")] = Path("fund.toml"),
) -> None:
    """Show portfolios, the halt flag, the last cycle, and today's model spend."""
    config = load_config(config_path)
    journal = _journal(config)
    chosen = _mode(mode)
    halt = journal.halt_state()
    cycles = journal.load_cycles()
    console.print(f"Mode {chosen}.")
    if halt is None:
        console.print("Halt: clear.")
    else:
        console.print(f"Halted. {halt.get('reason', '')} at {halt.get('at', '')}.")
    if cycles:
        last = cycles[-1]
        stamp = last.finished_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
        console.print(f"Last cycle {last.cycle_id} finished {stamp} in {last.mode}.")
    else:
        console.print("Last cycle: none.")
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    console.print(f"Model spend today: ${journal.llm_spend_on(today):.2f}.")
    table = Table(title="Positions")
    table.add_column("Portfolio")
    table.add_column("Market")
    table.add_column("Size", justify="right")
    table.add_column("Entry", justify="right")
    rows = 0
    for portfolio_id in _portfolio_ids(config):
        account = journal.load_paper(portfolio_id)
        if account is None:
            continue
        for position in account.positions.values():
            if position.size == 0:
                continue
            table.add_row(portfolio_id, position.symbol, f"{position.size:.6f}", f"{position.entry_px:.2f}")
            rows += 1
    if rows:
        console.print(table)
    else:
        console.print("No open positions.")


@app.command()
def score(
    config_path: Annotated[Path, typer.Option("--config", help="Path to fund.toml.")] = Path("fund.toml"),
) -> None:
    """Build scores.json and print the leaderboard."""
    config = load_config(config_path)
    journal = _journal(config)
    report = build_report(journal, datetime.now(UTC))
    journal.write_scores(report)
    table = Table(title="Leaderboard")
    for heading in ("Portfolio", "Kind", "Return", "Net of model cost", "Sharpe", "Cycles"):
        table.add_column(heading, justify="right" if heading != "Portfolio" and heading != "Kind" else "left")
    ordered = sorted(report.portfolios, key=lambda item: (-item.return_net_of_llm, item.portfolio_id))
    for item in ordered:
        sharpe = "n/a" if item.sharpe is None else f"{item.sharpe:.2f}"
        table.add_row(
            item.portfolio_id,
            item.kind,
            f"{item.total_return * 100:+.2f}%",
            f"{item.return_net_of_llm * 100:+.2f}%",
            sharpe,
            str(item.cycles),
        )
    if ordered:
        console.print(table)
    else:
        console.print("No scores yet.")
    console.print(f"Wrote {journal.root / 'scores.json'}.")


@app.command()
def dashboard(
    out: Annotated[Path, typer.Option("--out", help="Directory for the static site.")] = Path("site"),
    config_path: Annotated[Path, typer.Option("--config", help="Path to fund.toml.")] = Path("fund.toml"),
) -> None:
    """Render the static research site."""
    config = load_config(config_path)
    build_site(_journal(config), config, out, datetime.now(UTC))
    console.print(f"Wrote {out / 'index.html'}.")


@app.command()
def halt(
    flatten: Annotated[bool, typer.Option("--flatten", help="Close live positions and stops.")] = False,
    mode: Annotated[str | None, typer.Option("--mode", help="paper, testnet, or live.")] = None,
    config_path: Annotated[Path, typer.Option("--config", help="Path to fund.toml.")] = Path("fund.toml"),
) -> None:
    """Set the halt flag. With --flatten, close the live book."""
    config = load_config(config_path)
    secrets = load_secrets()
    journal = _journal(config)
    now = datetime.now(UTC)
    journal.set_halt("manual", now)
    console.print("Halt set.")
    if not flatten:
        return
    chosen = _mode(mode)
    if chosen == "paper":
        console.print("Paper mode has no live book to flatten.")
        return
    venue = _venue(config, secrets, chosen)
    account = venue.account()
    for symbol, position in account.positions.items():
        if position.size != 0:
            outcome = venue.close_position(symbol)
            console.print(f"Close {symbol}: {outcome.status}.")
        venue.sync_stop(symbol, None)
    console.print("Live book flattened.")


@app.command()
def resume(
    config_path: Annotated[Path, typer.Option("--config", help="Path to fund.toml.")] = Path("fund.toml"),
) -> None:
    """Clear the halt flag after a confirmation."""
    config = load_config(config_path)
    journal = _journal(config)
    typer.confirm("Clear the halt and allow the next cycle to trade?", abort=True)
    journal.clear_halt()
    console.print("Halt cleared.")


@app.command()
def doctor(
    config_path: Annotated[Path, typer.Option("--config", help="Path to fund.toml.")] = Path("fund.toml"),
) -> None:
    """Check keys, the journal, market data, the live account, and each Claude model."""
    config = load_config(config_path)
    secrets = load_secrets()
    rows = _doctor(config, secrets)
    table = Table(title="Doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail")
    failed = False
    for name, ok, detail in rows:
        table.add_row(name, "pass" if ok else "fail", detail)
        failed = failed or not ok
    console.print(table)
    if failed:
        raise typer.Exit(1)


@app.command()
def replay(
    snapshot_path: Annotated[Path, typer.Argument(help="Saved snapshot JSON.")],
    config_id: Annotated[str | None, typer.Option("--config-id", help="Defaults to the champion.")] = None,
    times: Annotated[int, typer.Option("--times", help="How many times to repeat the same snapshot.")] = 3,
    config_path: Annotated[Path, typer.Option("--config", help="Path to fund.toml.")] = Path("fund.toml"),
) -> None:
    """Run a decision and the risk gate on a saved snapshot. No orders are sent."""
    config = load_config(config_path)
    secrets = load_secrets()
    if not secrets.anthropic_api_key:
        console.print("ANTHROPIC_API_KEY is not set. Add it to .env before replaying.")
        raise typer.Exit(1)
    snapshot = Snapshot.model_validate_json(snapshot_path.read_text())
    chosen = _agent(config, config_id)
    claude = Claude()
    journal = _journal(config)
    for index in range(1, times + 1):
        console.print(f"Replay {index} of {times}.")
        _replay_once(config, journal, claude, snapshot, chosen)


def _print_run(record: CycleRecord) -> None:
    champion = record.runs[0] if record.runs else None
    console.print(f"Cycle {record.cycle_id} ({record.mode}).")
    if champion is None or champion.gate is None:
        console.print("No champion gate result.")
        return
    table = Table(title=champion.config_id)
    for heading in ("Market", "Requested", "Approved", "Status", "Reasons"):
        justify = "left" if heading in {"Market", "Status", "Reasons"} else "right"
        table.add_column(heading, justify=justify)
    for target in champion.gate.targets:
        reasons = ", ".join(target.reasons) if target.reasons else ""
        table.add_row(
            target.symbol,
            f"{target.requested_weight:+.2f}",
            f"{target.approved_weight:+.2f}",
            target.status,
            reasons,
        )
    console.print(table)
    cost = sum(item.cost_usd for item in champion.usage)
    console.print(f"Orders {len(champion.outcomes)}. Cost ${cost:.4f}.")


def _replay_once(
    config: FundConfig,
    journal: Journal,
    claude: Claude,
    snapshot: Snapshot,
    agent: AgentConfig,
) -> None:
    portfolio = _flat_portfolio(config, snapshot)
    track = journal.track_record(agent.config_id, snapshot)
    result = asyncio.run(decide(agent, claude, snapshot, portfolio, track, config.risk))
    if result.error or result.decision is None:
        console.print(result.error or "The model returned no decision.")
        return
    gate = apply_gate(result.decision, portfolio, snapshot, config.universe, config.risk, snapshot.as_of)
    table = Table(title=agent.config_id)
    table.add_column("Market")
    table.add_column("Probability", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Approved", justify="right")
    table.add_column("Status")
    approved = {item.symbol: item for item in gate.targets}
    for item in result.decision.decisions:
        gated = approved.get(item.symbol)
        table.add_row(
            item.symbol,
            f"{item.prob_up_24h:.2f}",
            f"{item.target_weight:+.2f}",
            "n/a" if gated is None else f"{gated.approved_weight:+.2f}",
            "n/a" if gated is None else gated.status,
        )
    console.print(result.decision.market_view)
    console.print(table)
    tokens = sum(item.input_tokens + item.output_tokens for item in result.usage)
    cost = sum(item.cost_usd for item in result.usage)
    console.print(f"Tokens {tokens}. Cost ${cost:.4f}.")


def _flat_portfolio(config: FundConfig, snapshot: Snapshot) -> PortfolioState:
    equity = config.fund.paper_starting_equity
    return PortfolioState(
        portfolio_id=f"config:{config.champion.config_id}",
        as_of=snapshot.as_of,
        account=AccountSnapshot(equity=equity, positions={}),
        bookkeeping=Bookkeeping(
            day=snapshot.as_of.astimezone(UTC).strftime("%Y-%m-%d"),
            day_start_equity=equity,
            high_water_mark=equity,
        ),
    )


def _doctor(config: FundConfig, secrets: Secrets) -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []
    rows.append(_check_key(secrets))
    rows.append(_check_journal(config))
    rows.append(_check_markets(config))
    rows.append(_check_account(config, secrets))
    rows.extend(_check_models(config, secrets))
    return rows


def _check_key(secrets: Secrets) -> tuple[str, bool, str]:
    if secrets.anthropic_api_key:
        return ("ANTHROPIC_API_KEY", True, "set")
    return ("ANTHROPIC_API_KEY", False, "Add ANTHROPIC_API_KEY to .env.")


def _check_journal(config: FundConfig) -> tuple[str, bool, str]:
    root = Path(config.fund.journal_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".doctor"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return ("journal", False, str(exc))
    return ("journal", True, str(root))


def _check_markets(config: FundConfig) -> tuple[str, bool, str]:
    try:
        quotes = HyperliquidData().quotes([item.symbol for item in config.universe])
    except Exception as exc:
        logger.exception("Market check failed.")
        return ("mainnet markets", False, str(exc))
    missing = [item.symbol for item in config.universe if item.symbol not in quotes]
    if missing:
        return ("mainnet markets", False, "Missing " + ", ".join(missing) + ".")
    return ("mainnet markets", True, f"{len(quotes)} symbols")


def _check_account(config: FundConfig, secrets: Secrets) -> tuple[str, bool, str]:
    if not secrets.hl_account_address or not secrets.hl_agent_private_key_testnet:
        return ("live account", True, "skipped, testnet key is not set")
    try:
        account = _venue(config, secrets, "testnet").account()
    except Exception as exc:
        logger.exception("Account check failed.")
        return ("live account", False, str(exc))
    return ("live account", True, f"equity {account.equity:.2f}")


def _check_models(config: FundConfig, secrets: Secrets) -> list[tuple[str, bool, str]]:
    if not secrets.anthropic_api_key:
        return [(f"model {key}", False, "ANTHROPIC_API_KEY is not set.") for key in config.catalog]
    claude = Claude()
    rows: list[tuple[str, bool, str]] = []
    for key, model in config.catalog.items():
        result = asyncio.run(
            claude.parse(
                purpose="doctor",
                model=model.model_copy(update={"max_tokens": 256}),
                system="Reply with the structured answer.",
                user="Say ok.",
                output_type=_Ping,
            )
        )
        if result.error or result.output is None:
            rows.append((f"model {key}", False, result.error or "no parsed output"))
        else:
            rows.append((f"model {key}", True, result.output.answer))
    return rows


class _Ping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


def _confirm(gate: GateResult, plans: list[OrderPlan]) -> bool:
    table = Table(title="Live orders")
    table.add_column("Market")
    table.add_column("Approved", justify="right")
    table.add_column("Status")
    for target in gate.targets:
        table.add_row(target.symbol, f"{target.approved_weight:+.2f}", target.status)
    console.print(table)
    if plans:
        for plan in plans:
            console.print(f"{plan.side} {plan.size} {plan.symbol} ({plan.reason}).")
    else:
        console.print("No live orders.")
    return typer.confirm("Send these orders?", default=False)


def _venue(config: FundConfig, secrets: Secrets, mode: Mode) -> HyperliquidVenue:
    if mode == "live":
        key = secrets.hl_agent_private_key_mainnet
        key_name = "HL_AGENT_PRIVATE_KEY_MAINNET"
        network = "mainnet"
        slippage = config.execution.mainnet_slippage
    else:
        key = secrets.hl_agent_private_key_testnet
        key_name = "HL_AGENT_PRIVATE_KEY_TESTNET"
        network = "testnet"
        slippage = config.execution.testnet_slippage
    if not secrets.hl_account_address:
        raise typer.BadParameter("HL_ACCOUNT_ADDRESS is not set. Add the dedicated wallet address to .env.")
    if not key:
        message = (
            f"{key_name} is not set. Run scripts/setup_agent.py --network {network} and add the key to .env."
        )
        raise typer.BadParameter(message)
    return HyperliquidVenue(
        network=network,
        account_address=secrets.hl_account_address,
        agent_private_key=key,
        symbols=[item.symbol for item in config.universe],
        slippage=slippage,
        allow_mainnet=secrets.allow_mainnet,
    )


def _agent(config: FundConfig, config_id: str | None) -> AgentConfig:
    chosen = config_id or config.champion.config_id
    for item in config.configs:
        if item.config_id == chosen:
            return item
    raise typer.BadParameter(f"Unknown config id {chosen}.")


def _mode(mode: str | None) -> Mode:
    raw = (mode or os.environ.get("FUND_MODE") or "paper").strip()
    if raw == "paper" or raw == "testnet" or raw == "live":
        return raw
    raise typer.BadParameter("Mode must be paper, testnet, or live.")


def _journal(config: FundConfig) -> Journal:
    return Journal(Path(config.fund.journal_dir))


def _portfolio_ids(config: FundConfig) -> list[str]:
    from fund.baselines import BASELINES

    ids = [f"config:{item.config_id}" for item in config.configs]
    ids.extend(f"baseline:{name}" for name in BASELINES)
    return ids


if __name__ == "__main__":
    app()
