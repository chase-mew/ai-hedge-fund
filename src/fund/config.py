"""Load fund.toml and environment secrets."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from fund.contracts import AssetClass, Frozen, MarketSpec

Effort = Literal["low", "medium", "high", "xhigh", "max"]
Style = Literal["single", "team"]
Autonomy = Literal["paper", "approve", "auto"]


class ModelSpec(Frozen):
    key: str
    id: str
    effort: Effort | None
    summarized_thinking: bool
    max_tokens: int
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float
    cache_read_per_mtok: float


class AgentConfig(Frozen):
    config_id: str
    model: ModelSpec
    style: Style


class RiskPolicy(Frozen):
    max_weight_per_market: float
    max_gross: float
    max_net: float
    max_new_risk_per_cycle: float
    min_trade_weight: float
    daily_loss_limit: float
    max_drawdown: float
    flip_cooldown_hours: float
    pre_weekend_max_weight: float
    max_mark_oracle_deviation: float
    stop_min_pct: float
    stop_max_pct: float
    default_stop_pct: float
    class_caps: dict[AssetClass, float]


class ExecutionSettings(Frozen):
    exchange_leverage: int
    min_order_usd: float
    mainnet_slippage: float
    testnet_slippage: float
    max_price_gap: float
    max_live_equity_usd: float


class PaperSettings(Frozen):
    slippage_bps: float
    core_taker_fee: float
    xyz_taker_fee: float


class NewsSettings(Frozen):
    enabled: bool
    model: str
    max_searches: int
    tool_type: str
    search_cost_usd: float


class FundSettings(Frozen):
    name: str
    cycle_hours: int
    paper_starting_equity: float
    daily_llm_budget_usd: float
    journal_dir: str
    autonomy: Autonomy


class FundConfig(Frozen):
    fund: FundSettings
    champion: AgentConfig
    shadows: list[AgentConfig]
    catalog: dict[str, ModelSpec]
    news: NewsSettings
    risk: RiskPolicy
    execution: ExecutionSettings
    paper: PaperSettings
    universe: list[MarketSpec]

    @property
    def configs(self) -> list[AgentConfig]:
        return [self.champion, *self.shadows]


class Secrets(Frozen):
    anthropic_api_key: str | None = None
    hl_account_address: str | None = None
    hl_agent_private_key_testnet: str | None = None
    hl_agent_private_key_mainnet: str | None = None
    allow_mainnet: bool = False

    def __repr__(self) -> str:
        return (
            "Secrets("
            f"anthropic_api_key={'set' if self.anthropic_api_key else 'missing'}, "
            f"hl_account_address={'set' if self.hl_account_address else 'missing'}, "
            f"hl_agent_private_key_testnet={'set' if self.hl_agent_private_key_testnet else 'missing'}, "
            f"hl_agent_private_key_mainnet={'set' if self.hl_agent_private_key_mainnet else 'missing'}, "
            f"allow_mainnet={self.allow_mainnet})"
        )


def load_config(path: Path = Path("fund.toml")) -> FundConfig:
    """Read fund.toml and validate cross references."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    catalog = _catalog(raw["models"]["catalog"])
    champion = _agent(str(raw["models"]["champion"]), catalog)
    shadows = [_agent(str(item), catalog) for item in raw["models"]["shadows"]]
    shadow_ids = [item.config_id for item in shadows]
    if champion.config_id in shadow_ids:
        raise ValueError(f"Champion {champion.config_id} is also listed as a shadow.")
    if len(shadow_ids) != len(set(shadow_ids)):
        raise ValueError("Shadow config ids must be unique.")
    news_model = str(raw["news"]["model"])
    if news_model not in catalog:
        raise ValueError(f"News model {news_model} is not in the model catalog.")
    universe = [MarketSpec.model_validate(item) for item in raw["universe"]]
    symbols = [item.symbol for item in universe]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Universe symbols must be unique.")
    risk = _risk(raw["risk"])
    missing_caps = sorted({item.asset_class for item in universe if item.asset_class not in risk.class_caps})
    if missing_caps:
        names = ", ".join(item.value for item in missing_caps)
        raise ValueError(f"Missing class cap for: {names}.")
    return FundConfig(
        fund=FundSettings.model_validate(raw["fund"]),
        champion=champion,
        shadows=shadows,
        catalog=catalog,
        news=NewsSettings.model_validate(raw["news"]),
        risk=risk,
        execution=ExecutionSettings.model_validate(raw["execution"]),
        paper=PaperSettings.model_validate(raw["paper"]),
        universe=universe,
    )


def load_secrets() -> Secrets:
    """Load optional secrets from the environment. Missing values stay None."""
    load_dotenv()
    return Secrets(
        anthropic_api_key=_optional("ANTHROPIC_API_KEY"),
        hl_account_address=_optional("HL_ACCOUNT_ADDRESS"),
        hl_agent_private_key_testnet=_optional("HL_AGENT_PRIVATE_KEY_TESTNET"),
        hl_agent_private_key_mainnet=_optional("HL_AGENT_PRIVATE_KEY_MAINNET"),
        allow_mainnet=os.environ.get("FUND_ALLOW_MAINNET", "").strip() == "yes",
    )


def _optional(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _catalog(raw: dict[str, dict[str, object]]) -> dict[str, ModelSpec]:
    catalog: dict[str, ModelSpec] = {}
    for key, fields in raw.items():
        effort = fields.get("effort")
        if effort == "":
            effort = None
        catalog[key] = ModelSpec.model_validate({**fields, "key": key, "effort": effort})
    return catalog


def _agent(config_id: str, catalog: dict[str, ModelSpec]) -> AgentConfig:
    parts = config_id.split("/")
    if len(parts) != 2:
        raise ValueError(f"Config id {config_id} must look like '<model key>/single' or '<model key>/team'.")
    key, raw_style = parts
    if raw_style == "single":
        style: Style = "single"
    elif raw_style == "team":
        style = "team"
    else:
        raise ValueError(f"Config id {config_id} must look like '<model key>/single' or '<model key>/team'.")
    model = catalog.get(key)
    if model is None:
        raise ValueError(f"Config id {config_id} uses unknown model {key}.")
    return AgentConfig(config_id=config_id, model=model, style=style)


def _risk(raw: dict[str, object]) -> RiskPolicy:
    caps_raw = raw.get("class_caps")
    if not isinstance(caps_raw, dict):
        raise ValueError("risk.class_caps must be a table of asset class caps.")
    caps = {AssetClass(str(key)): float(value) for key, value in caps_raw.items()}
    fields = {key: value for key, value in raw.items() if key != "class_caps"}
    return RiskPolicy.model_validate({**fields, "class_caps": caps})
