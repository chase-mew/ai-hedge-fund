# pyright: strict
"""Journal of cycles, forecasts, equity and account state."""

from __future__ import annotations

import gzip
import json
import os
import re
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fund.clock import utc_day
from fund.contracts import (
    AccountSnapshot,
    Bookkeeping,
    Candle,
    CycleRecord,
    EquityPoint,
    Forecast,
    Frozen,
    PaperAccount,
    RecentDecision,
    Resolution,
    ScoreReport,
    Snapshot,
    TrackRecord,
)
from fund.scoring.metrics import brier

_CYCLE_ID = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})Z$")
_HOUR = timedelta(hours=1)


class _Halt(Frozen):
    reason: str
    at: str


def update_bookkeeping(bookkeeping: Bookkeeping | None, equity: float, now: datetime) -> Bookkeeping:
    """Reset day start equity on a new UTC day and raise the high water mark."""
    day = utc_day(now)
    if bookkeeping is None:
        return Bookkeeping(day=day, day_start_equity=equity, high_water_mark=equity)
    mark = max(bookkeeping.high_water_mark, equity)
    if bookkeeping.day != day:
        return Bookkeeping(
            day=day,
            day_start_equity=equity,
            high_water_mark=mark,
            last_flip_at=dict(bookkeeping.last_flip_at),
        )
    return bookkeeping.model_copy(
        update={"high_water_mark": mark, "last_flip_at": dict(bookkeeping.last_flip_at)}
    )


def record_flips(
    bookkeeping: Bookkeeping,
    before: AccountSnapshot,
    after: AccountSnapshot,
    now: datetime,
) -> Bookkeeping:
    """Stamp last flip time when a position changes sign."""
    flips = dict(bookkeeping.last_flip_at)
    symbols = set(before.positions) | set(after.positions)
    for symbol in symbols:
        if _signed_size(before, symbol) * _signed_size(after, symbol) < 0.0:
            flips[symbol] = now
    return bookkeeping.model_copy(update={"last_flip_at": flips})


def resolve(forecast: Forecast, hourly: list[Candle], now: datetime) -> Resolution | None:
    """Resolve a forecast from the hourly close once an hour has passed."""
    if now < forecast.resolves_at + _HOUR:
        return None
    opened = forecast.resolves_at.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    for candle in hourly:
        if candle.open_time.astimezone(UTC) != opened:
            continue
        return Resolution(
            forecast_id=forecast.forecast_id,
            resolved_at=now,
            price=candle.close,
            outcome=candle.close > forecast.ref_price,
        )
    return None


class Journal:
    """Files under a journal root."""

    def __init__(self, root: Path) -> None:
        """Remember the directory that holds this journal."""
        self.root = root

    def has_cycle(self, cycle_id: str) -> bool:
        """Return whether this cycle id is already stored."""
        return _cycle_path(self.root, cycle_id).is_file()

    def write_cycle(self, record: CycleRecord) -> Path:
        """Store a cycle record and return its path."""
        path = _cycle_path(self.root, record.cycle_id)
        _write_model(path, record)
        return path

    def write_transcripts(self, cycle_id: str, config_id: str, transcripts: list[dict[str, Any]]) -> str:
        """Gzip transcripts for one config and return the relative path."""
        path = _transcript_path(self.root, cycle_id, config_id)
        _atomic_write(path, gzip.compress(json.dumps(transcripts).encode()))
        return path.relative_to(self.root).as_posix()

    def load_cycles(self, limit: int | None = None) -> list[CycleRecord]:
        """Return stored cycles oldest first, keeping the newest when limit is set."""
        folder = self.root / "cycles"
        if not folder.exists():
            return []
        records = [CycleRecord.model_validate_json(path.read_text()) for path in folder.rglob("*.json")]
        records.sort(key=lambda record: (record.started_at, record.cycle_id))
        if limit is None:
            return records
        if limit <= 0:
            return []
        return records[-limit:]

    def append_forecasts(self, rows: list[Forecast]) -> None:
        """Append forecast rows."""
        _append_models(self.root / "forecasts.jsonl", rows)

    def append_resolutions(self, rows: list[Resolution]) -> None:
        """Append resolution rows."""
        _append_models(self.root / "resolutions.jsonl", rows)

    def unresolved_forecasts(self, now: datetime) -> list[Forecast]:
        """Return forecasts that are due and still unresolved."""
        done = {row.forecast_id for row in _load_jsonl(self.root / "resolutions.jsonl", Resolution)}
        return [
            forecast
            for forecast in _load_jsonl(self.root / "forecasts.jsonl", Forecast)
            if forecast.forecast_id not in done and now >= forecast.resolves_at + _HOUR
        ]

    def append_equity(self, rows: list[EquityPoint]) -> None:
        """Append equity points."""
        _append_models(self.root / "equity.jsonl", rows)

    def load_equity(self) -> list[EquityPoint]:
        """Return equity points in append order."""
        return _load_jsonl(self.root / "equity.jsonl", EquityPoint)

    def load_paper(self, portfolio_id: str) -> PaperAccount | None:
        """Load a paper account, or None if it has not been saved."""
        return _read_model(self.root / "state" / "paper" / f"{_slug(portfolio_id)}.json", PaperAccount)

    def save_paper(self, account: PaperAccount) -> None:
        """Store a paper account under its portfolio slug."""
        _write_model(self.root / "state" / "paper" / f"{_slug(account.portfolio_id)}.json", account)

    def load_live_bookkeeping(self) -> Bookkeeping | None:
        """Load live bookkeeping, or None if it has not been saved."""
        return _read_model(self.root / "state" / "live.json", Bookkeeping)

    def save_live_bookkeeping(self, bookkeeping: Bookkeeping) -> None:
        """Store live account bookkeeping."""
        _write_model(self.root / "state" / "live.json", bookkeeping)

    def halt_state(self) -> dict[str, str] | None:
        """Return the halt reason and time, or None when trading is allowed."""
        halt = _read_model(self.root / "state" / "halt.json", _Halt)
        if halt is None:
            return None
        return {"reason": halt.reason, "at": halt.at}

    def set_halt(self, reason: str, now: datetime) -> None:
        """Record a halt reason until it is cleared."""
        stamp = now.astimezone(UTC).isoformat()
        _write_model(self.root / "state" / "halt.json", _Halt(reason=reason, at=stamp))

    def clear_halt(self) -> None:
        """Remove the halt flag."""
        (self.root / "state" / "halt.json").unlink(missing_ok=True)

    def llm_spend_on(self, day: str) -> float:
        """Sum LLM costs for cycles that started on this UTC day."""
        total = 0.0
        for record in self.load_cycles():
            if utc_day(record.started_at) != day:
                continue
            if record.news_usage is not None:
                total += record.news_usage.cost_usd
            for run in record.runs:
                total += sum(item.cost_usd for item in run.usage)
        return total

    def track_record(self, config_id: str, snapshot: Snapshot, cycles: int = 3) -> TrackRecord:
        """Recent nonzero decisions for a config, plus its resolved Brier score."""
        current = {market.symbol: market.mark for market in snapshot.markets}
        matching = [record for record in self.load_cycles() if _ran(record, config_id)]
        window = matching[-cycles:] if cycles > 0 else []
        recent: list[RecentDecision] = []
        for record in window:
            recent.extend(_recent_rows(record, config_id, current))
        probs, outcomes = _resolved_pairs(self.root, config_id)
        return TrackRecord(recent=recent, resolved_forecasts=len(probs), brier=brier(probs, outcomes))

    def write_scores(self, report: ScoreReport) -> None:
        """Store the latest score report."""
        _write_model(self.root / "scores.json", report)

    def load_scores(self) -> ScoreReport | None:
        """Load the score report, or None if scores have not been written."""
        return _read_model(self.root / "scores.json", ScoreReport)


def _load_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    if not path.is_file():
        return []
    return [model.model_validate_json(line) for line in path.read_text().splitlines()]


def _signed_size(account: AccountSnapshot, symbol: str) -> float:
    position = account.positions.get(symbol)
    if position is None:
        return 0.0
    return position.size


def _ran(record: CycleRecord, config_id: str) -> bool:
    return any(run.config_id == config_id for run in record.runs)


def _recent_rows(record: CycleRecord, config_id: str, current: dict[str, float]) -> list[RecentDecision]:
    then = {market.symbol: market.mark for market in record.snapshot.markets}
    rows: list[RecentDecision] = []
    for run in record.runs:
        if run.config_id != config_id or run.decision is None:
            continue
        for decision in run.decision.decisions:
            if decision.target_weight == 0.0:
                continue
            price_then = then.get(decision.symbol, 0.0)
            rows.append(
                RecentDecision(
                    cycle_id=record.cycle_id,
                    symbol=decision.symbol,
                    target_weight=decision.target_weight,
                    prob_up_24h=decision.prob_up_24h,
                    thesis=decision.thesis,
                    price_then=price_then,
                    price_now=current.get(decision.symbol, price_then),
                )
            )
    return rows


def _resolved_pairs(root: Path, config_id: str) -> tuple[list[float], list[bool]]:
    resolutions = {row.forecast_id: row for row in _load_jsonl(root / "resolutions.jsonl", Resolution)}
    probs: list[float] = []
    outcomes: list[bool] = []
    for forecast in _load_jsonl(root / "forecasts.jsonl", Forecast):
        if forecast.config_id != config_id:
            continue
        resolution = resolutions.get(forecast.forecast_id)
        if resolution is None:
            continue
        probs.append(forecast.prob_up)
        outcomes.append(resolution.outcome)
    return probs, outcomes


def _slug(value: str) -> str:
    return value.replace("/", "__").replace(":", "__")


def _cycle_stamp(cycle_id: str) -> datetime:
    match = _CYCLE_ID.fullmatch(cycle_id)
    if match is None:
        raise ValueError(f"Cycle id {cycle_id!r} is not a UTC slot.")
    try:
        return datetime(
            int(match[1]),
            int(match[2]),
            int(match[3]),
            int(match[4]),
            int(match[5]),
            tzinfo=UTC,
        )
    except ValueError as exc:
        raise ValueError(f"Cycle id {cycle_id!r} is not a UTC slot.") from exc


def _cycle_path(root: Path, cycle_id: str) -> Path:
    stamp = _cycle_stamp(cycle_id)
    return root / "cycles" / f"{stamp:%Y}" / f"{stamp:%m}" / f"{stamp:%d}" / f"{stamp:%H%M}.json"


def _transcript_path(root: Path, cycle_id: str, config_id: str) -> Path:
    stamp = _cycle_stamp(cycle_id)
    name = f"{_slug(config_id)}.json.gz"
    return root / "transcripts" / f"{stamp:%Y}" / f"{stamp:%m}" / f"{stamp:%d}" / f"{stamp:%H%M}" / name


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_model(path: Path, model: BaseModel) -> None:
    _atomic_write(path, model.model_dump_json().encode())


def _read_model[T: BaseModel](path: Path, model: type[T]) -> T | None:
    if not path.is_file():
        return None
    return model.model_validate_json(path.read_text())


def _append_models[T: BaseModel](path: Path, rows: list[T]) -> None:
    if not rows:
        return
    existing = path.read_bytes() if path.is_file() else b""
    if existing and not existing.endswith(b"\n"):
        existing += b"\n"
    payload = b"".join(row.model_dump_json().encode() + b"\n" for row in rows)
    _atomic_write(path, existing + payload)
