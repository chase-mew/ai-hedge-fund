# pyright: strict
"""Shared data contracts. Returns, rates and weights are fractions (0.01 means 1%)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AssetClass(StrEnum):
    EQUITY_INDEX = "equity_index"
    STOCK = "stock"
    COMMODITY = "commodity"
    FX = "fx"
    CRYPTO = "crypto"


Mode = Literal["paper", "testnet", "live"]
Side = Literal["buy", "sell"]


# Market data


class MarketSpec(Frozen):
    symbol: str
    display_name: str
    asset_class: AssetClass
    tradfi_hours: bool


class MarketMeta(Frozen):
    symbol: str
    sz_decimals: int
    max_leverage: int
    only_isolated: bool


class Candle(Frozen):
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketQuote(Frozen):
    symbol: str
    mark: float
    oracle: float
    prev_day_px: float
    funding_hourly: float
    open_interest: float
    day_volume_usd: float


class SessionState(Frozen):
    tradfi_open: bool
    pre_weekend: bool


class MarketFeatures(Frozen):
    symbol: str
    display_name: str
    asset_class: AssetClass
    tradfi_hours: bool
    market_open: bool
    mark: float
    oracle: float
    ret_24h: float
    ret_7d: float
    ret_30d: float
    vol_30d_ann: float
    pct_from_20d_high: float
    pct_from_20d_low: float
    sma20_gap: float
    sma50_gap: float
    funding_ann: float
    open_interest_usd: float
    volume_24h_usd: float
    daily_closes: list[float]


class NewsDigest(Frozen):
    text: str
    sources: list[str]
    compiled_at: datetime


class Snapshot(Frozen):
    cycle_id: str
    as_of: datetime
    session: SessionState
    markets: list[MarketFeatures]
    news: NewsDigest | None = None


# Portfolio


class Position(Frozen):
    symbol: str
    size: float
    entry_px: float
    mark: float

    @property
    def notional(self) -> float:
        return self.size * self.mark


class AccountSnapshot(Frozen):
    equity: float
    positions: dict[str, Position]


class Bookkeeping(Frozen):
    day: str
    day_start_equity: float
    high_water_mark: float
    last_flip_at: dict[str, datetime] = Field(default_factory=dict)


class PortfolioState(Frozen):
    portfolio_id: str
    as_of: datetime
    account: AccountSnapshot
    bookkeeping: Bookkeeping

    @property
    def equity(self) -> float:
        return self.account.equity

    def weight(self, symbol: str) -> float:
        position = self.account.positions.get(symbol)
        if position is None or self.account.equity <= 0:
            return 0.0
        return position.notional / self.account.equity


class RecentDecision(Frozen):
    cycle_id: str
    symbol: str
    target_weight: float
    prob_up_24h: float
    thesis: str
    price_then: float
    price_now: float


class TrackRecord(Frozen):
    recent: list[RecentDecision]
    resolved_forecasts: int
    brier: float | None


# Claude output. Field order is the order Claude writes them, so reasoning comes before numbers.


class MarketDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(description="Market symbol exactly as listed in the snapshot, for example xyz:SP500.")
    thesis: str = Field(description="One or two sentences grounded in the snapshot data or the news digest.")
    invalidation: str = Field(description="The observable condition that would prove the thesis wrong.")
    prob_up_24h: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated probability that the mark price is higher 24 hours from now. 0.5 means no view."
        ),
    )
    target_weight: float = Field(
        ge=-0.25,
        le=0.25,
        description=(
            "Target signed exposure as a fraction of equity. "
            "0.10 is long 10% of equity, -0.10 is short 10%, 0 is flat."
        ),
    )
    stop_loss_pct: float = Field(
        ge=0.5,
        le=20.0,
        description="Stop distance from entry price in percent. Use a wider stop for more volatile markets.",
    )


class PortfolioDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_view: str = Field(description="Two to four sentences on the overall market regime.")
    decisions: list[MarketDecision] = Field(description="Exactly one entry for every market in the universe.")


class AnalystView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    key_points: list[str] = Field(description="Two to four short evidence points from the data or news.")
    risks: list[str] = Field(description="One to three risks to the view.")
    stance: Literal["bullish", "neutral", "bearish"]
    prob_up_24h: float = Field(ge=0.0, le=1.0)


class AnalystReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Two to four sentences on this asset class.")
    views: list[AnalystView] = Field(description="Exactly one entry for every market assigned to you.")


# Risk gate


GateStatus = Literal["approved", "clipped", "held", "rejected"]


class GatedTarget(Frozen):
    symbol: str
    current_weight: float
    requested_weight: float
    approved_weight: float
    stop_loss_pct: float
    status: GateStatus
    reasons: list[str]


class GateResult(Frozen):
    targets: list[GatedTarget]
    halt: bool = False
    halt_reason: str | None = None


# Execution


class OrderPlan(Frozen):
    symbol: str
    side: Side
    size: float
    reduce_only: bool
    close_all: bool
    reason: Literal["reduce", "close", "flip_close", "flip_open", "increase"]


class OrderOutcome(Frozen):
    symbol: str
    side: Side
    requested_size: float
    filled_size: float
    avg_px: float | None
    fee_usd: float
    status: Literal["filled", "partial", "unfilled", "error", "skipped"]
    order_id: str | None = None
    message: str = ""


class StopOrder(Frozen):
    symbol: str
    side: Side
    size: float
    trigger_px: float


# Paper accounts


class PaperPosition(Frozen):
    symbol: str
    size: float
    entry_px: float
    stop: StopOrder | None = None


class PaperAccount(Frozen):
    portfolio_id: str
    cash: float
    positions: dict[str, PaperPosition]
    last_marks: dict[str, float]
    updated_at: datetime
    bookkeeping: Bookkeeping
    fees_cum: float = 0.0
    funding_cum: float = 0.0
    llm_cost_cum: float = 0.0


# Journal


class LlmUsage(Frozen):
    purpose: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    web_searches: int
    cost_usd: float


class ConfigRun(Frozen):
    config_id: str
    portfolio_id: str
    decision: PortfolioDecision | None
    error: str | None
    gate: GateResult | None
    outcomes: list[OrderOutcome]
    usage: list[LlmUsage]
    transcript_path: str | None
    equity_before: float
    equity_after: float


class CycleRecord(Frozen):
    cycle_id: str
    mode: Mode
    started_at: datetime
    finished_at: datetime
    snapshot: Snapshot
    news_usage: LlmUsage | None
    runs: list[ConfigRun]
    errors: list[str]


class Forecast(Frozen):
    forecast_id: str
    cycle_id: str
    config_id: str
    symbol: str
    made_at: datetime
    resolves_at: datetime
    prob_up: float
    ref_price: float


class Resolution(Frozen):
    forecast_id: str
    resolved_at: datetime
    price: float
    outcome: bool


class EquityPoint(Frozen):
    at: datetime
    cycle_id: str
    portfolio_id: str
    equity: float
    gross: float
    net: float
    fees_cum: float
    funding_cum: float
    llm_cost_cum: float


# Scores


class CalibrationBin(Frozen):
    lower: float
    upper: float
    mean_prob: float
    hit_rate: float
    count: int


class PortfolioScore(Frozen):
    portfolio_id: str
    kind: Literal["live", "config", "baseline"]
    cycles: int
    start_equity: float
    equity: float
    total_return: float
    return_net_of_llm: float
    ann_vol: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    fees: float
    funding: float
    llm_cost: float


class ForecastScore(Frozen):
    config_id: str
    resolved: int
    brier: float | None
    brier_skill: float | None
    calibration: list[CalibrationBin]


class ScoreReport(Frozen):
    generated_at: datetime
    portfolios: list[PortfolioScore]
    forecasts: list[ForecastScore]
