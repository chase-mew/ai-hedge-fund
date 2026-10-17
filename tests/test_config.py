"""Config loading and secret redaction."""

from pathlib import Path

import pytest

from fund.config import load_config, load_secrets
from fund.contracts import AssetClass


def test_load_config_resolves_champion_and_universe() -> None:
    config = load_config()
    assert config.fund.name == "AI Hedge Fund"
    assert config.champion.config_id == "opus-5.5/single"
    assert config.champion.style == "single"
    assert config.champion.model.id == "claude-opus-5-5"
    assert config.champion.model.key == "opus-5.5"
    assert [item.config_id for item in config.shadows] == [
        "opus-5.5/team",
        "sonnet-5.5/single",
        "haiku-5.5/single",
    ]
    assert config.configs[0].config_id == "opus-5.5/single"
    assert len(config.universe) == 11
    assert config.universe[1].display_name == "XYZ US 100"
    assert config.news.model == "sonnet-5.5"
    assert config.risk.class_caps[AssetClass.CRYPTO] == 0.4
    assert config.fund.autonomy == "approve"


def test_rejects_duplicate_symbols(tmp_path: Path) -> None:
    text = (
        Path("fund.toml")
        .read_text()
        .replace(
            'symbol = "ETH"\ndisplay_name = "Ether"',
            'symbol = "BTC"\ndisplay_name = "Ether"',
        )
    )
    path = tmp_path / "fund.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="unique"):
        load_config(path)


def test_rejects_champion_listed_as_shadow(tmp_path: Path) -> None:
    text = (
        Path("fund.toml")
        .read_text()
        .replace(
            'shadows = ["opus-5.5/team", "sonnet-5.5/single", "haiku-5.5/single"]',
            'shadows = ["opus-5.5/single"]',
        )
    )
    path = tmp_path / "fund.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="Champion"):
        load_config(path)


def test_rejects_unknown_model(tmp_path: Path) -> None:
    text = (
        Path("fund.toml")
        .read_text()
        .replace(
            'champion = "opus-5.5/single"',
            'champion = "missing/single"',
        )
    )
    path = tmp_path / "fund.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="unknown model"):
        load_config(path)


def test_rejects_missing_class_cap(tmp_path: Path) -> None:
    text = Path("fund.toml").read_text().replace("crypto = 0.4\n", "")
    path = tmp_path / "fund.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="crypto"):
        load_config(path)


def test_secrets_repr_hides_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    monkeypatch.setenv("HL_AGENT_PRIVATE_KEY_TESTNET", "0xabc123secret")
    monkeypatch.setenv("FUND_ALLOW_MAINNET", "yes")
    monkeypatch.delenv("HL_ACCOUNT_ADDRESS", raising=False)
    monkeypatch.delenv("HL_AGENT_PRIVATE_KEY_MAINNET", raising=False)
    secrets = load_secrets()
    rendered = repr(secrets)
    assert "sk-ant-secret-value" not in rendered
    assert "0xabc123secret" not in rendered
    assert secrets.allow_mainnet is True
    assert "set" in rendered
