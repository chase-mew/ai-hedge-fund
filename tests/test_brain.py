"""Brain client, costs, decisions, news, and the offline fake."""

import asyncio
from datetime import UTC, datetime
from typing import Any, cast

import httpx2
import pytest
from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic
from pydantic import BaseModel, ConfigDict, ValidationError

from fund.brain.costs import usage_cost
from fund.brain.decide import decide
from fund.brain.fake import FakeClaude
from fund.brain.llm import Claude, LlmResult
from fund.brain.news import compile_news
from fund.brain.prompts import SYSTEM_ANALYST, SYSTEM_PM, SYSTEM_TEAM_PM, render_limits
from fund.config import AgentConfig, ModelSpec, NewsSettings, load_config
from fund.contracts import AnalystReport, AnalystView, LlmUsage, PortfolioDecision, TrackRecord
from tests.factories import make_features, make_policy, make_portfolio, make_snapshot

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


class Counts:
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _Block:
    def __init__(self, kind: str, text: str = "", citations: list[dict[str, str]] | None = None) -> None:
        self.type = kind
        self.text = text
        self.citations = citations or []


class _SearchMessage:
    def __init__(self, stop_reason: str, content: list[_Block], usage: Any) -> None:
        self.stop_reason = stop_reason
        self.content = content
        self.usage = usage

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return {
            "stop_reason": self.stop_reason,
            "content": [
                {
                    "type": block.type,
                    "text": block.text,
                    "citations": block.citations,
                }
                for block in self.content
            ],
        }

    def to_param(self) -> dict[str, Any]:
        return {"role": "assistant", "content": self.model_dump()["content"]}


class _Parsed:
    def __init__(self, output: Any, stop_reason: str, usage: Any) -> None:
        self.parsed_output = output
        self.stop_reason = stop_reason
        self.usage = usage

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return {"stop_reason": self.stop_reason, "content": [{"type": "text", "text": "ok"}]}


def _usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write: int | None = 0,
    cache_read: int | None = 0,
    searches: int | None = 0,
) -> Any:
    server = None if searches is None else type("Server", (), {"web_search_requests": searches})()
    return type(
        "Usage",
        (),
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_write,
            "cache_read_input_tokens": cache_read,
            "server_tool_use": server,
        },
    )()


def _api_error(status: int, message: str) -> APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return APIStatusError(message, response=response, body=None)


def _validation_error() -> ValidationError:
    try:
        PortfolioDecision.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


def _scripted(items: list[Any]) -> tuple[AsyncAnthropic, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    class Messages:
        async def parse(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return _next(items)

        async def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return _next(items)

    class Client:
        def __init__(self) -> None:
            self.messages = Messages()

    return cast(AsyncAnthropic, Client()), calls


def _next(items: list[Any]) -> Any:
    item = items.pop(0)
    if isinstance(item, Exception):
        raise item
    return item


def _model() -> ModelSpec:
    return load_config().catalog["opus-5.5"]


def _news() -> NewsSettings:
    return load_config().news


def _single() -> AgentConfig:
    return AgentConfig(config_id="opus-5.5/single", model=_model(), style="single")


def _team() -> AgentConfig:
    return AgentConfig(config_id="opus-5.5/team", model=_model(), style="team")


def _track() -> TrackRecord:
    return TrackRecord(recent=[], resolved_forecasts=0, brier=None)


def _run_parse(
    client: AsyncAnthropic,
    *,
    output_type: type[Any] = PortfolioDecision,
    model: ModelSpec | None = None,
    system: str = "s",
    user: str = "u",
) -> LlmResult[Any]:
    return asyncio.run(
        Claude(client).parse(
            purpose="decision",
            model=model or _model(),
            system=system,
            user=user,
            output_type=output_type,
        )
    )


def _run_search(client: AsyncAnthropic, news: NewsSettings | None = None) -> LlmResult[str]:
    return asyncio.run(
        Claude(client).search(
            purpose="news",
            model=_model(),
            system="s",
            user="u",
            news=news or _news(),
        )
    )


def _run_news(
    claude: Any,
    *,
    settings: NewsSettings | None = None,
    universe: list[Any] | None = None,
    now: datetime = NOW,
) -> tuple[Any, Any, Any]:
    markets = load_config().universe if universe is None else universe
    return asyncio.run(compile_news(claude, _model(), settings or _news(), markets, now))


def _zero(purpose: str, model: ModelSpec) -> LlmUsage:
    return LlmUsage(
        purpose=purpose,
        model_id=model.id,
        input_tokens=0,
        output_tokens=0,
        cache_write_tokens=0,
        cache_read_tokens=0,
        web_searches=0,
        cost_usd=0.0,
    )


class Recording:
    def __init__(self, *, fail_groups: set[str] | None = None, error: str | None = None) -> None:
        self.fail_groups = fail_groups or set()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def parse(
        self,
        *,
        purpose: str,
        model: ModelSpec,
        system: str,
        user: str,
        output_type: type[Any],
    ) -> LlmResult[Any]:
        self.calls.append({"purpose": purpose, "system": system, "user": user, "output_type": output_type})
        usage = _zero(purpose, model)
        transcript: dict[str, Any] = {"request": {"purpose": purpose}, "response": None}
        if self.error is not None and output_type is PortfolioDecision:
            return LlmResult(None, usage, transcript, self.error)
        if output_type is AnalystReport:
            group = purpose.rsplit(":", 1)[-1]
            if group in self.fail_groups:
                return LlmResult(None, usage, transcript, "analyst broke")
            symbol = "BTC" if group == "crypto" else "xyz:GOLD" if group == "macro" else "xyz:SP500"
            report = AnalystReport(
                summary=f"{group} summary",
                views=[
                    AnalystView(
                        symbol=symbol,
                        key_points=["Point one."],
                        risks=["Risk one."],
                        stance="neutral",
                        prob_up_24h=0.5,
                    )
                ],
            )
            return LlmResult(report, usage, transcript, None)
        from tests.factories import make_decision

        return LlmResult(make_decision(), usage, transcript, None)

    async def search(
        self,
        *,
        purpose: str,
        model: ModelSpec,
        system: str,
        user: str,
        news: NewsSettings,
    ) -> LlmResult[str]:
        raise AssertionError(purpose)


def test_usage_cost_matches_hand_computed_rates() -> None:
    opus = load_config().catalog["opus-5.5"]
    # 1_500_000 * 4 + 250_000 * 20 + 100_000 * 5 + 4_000_000 * 0.40, per million, plus 3 * 0.01.
    assert usage_cost(Counts(1_500_000, 250_000, 100_000, 4_000_000), opus, 3, 0.01) == pytest.approx(13.13)
    haiku = load_config().catalog["haiku-5.5"]
    # 1_000_000 * (0.10 + 0.50 + 0.125 + 0.01) plus 1000 * 0.01.
    haiku_cost = usage_cost(Counts(1_000_000, 1_000_000, 1_000_000, 1_000_000), haiku, 1000, 0.01)
    assert haiku_cost == pytest.approx(10.735)


def test_usage_cost_is_zero_without_tokens_or_searches() -> None:
    assert usage_cost(Counts(0, 0, 0, 0), _model(), 0, 0.01) == 0.0


def test_parse_returns_a_decision() -> None:
    decision = PortfolioDecision.model_validate(
        {
            "market_view": "Quiet.",
            "decisions": [
                {
                    "symbol": "BTC",
                    "thesis": "No edge.",
                    "invalidation": "A gap through the average.",
                    "prob_up_24h": 0.5,
                    "target_weight": 0.0,
                    "stop_loss_pct": 5.0,
                }
            ],
        }
    )
    parsed = _Parsed(decision, "end_turn", _usage(1_000_000, cache_write=None, cache_read=None))
    client, calls = _scripted([parsed])
    result = _run_parse(client, system="system", user="hello")
    assert result.error is None
    assert result.output == decision
    assert result.usage.cost_usd == pytest.approx(4.0)
    assert result.usage.cache_write_tokens == 0
    assert result.usage.input_tokens == 1_000_000
    call = calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 16000
    assert call["output_format"] is PortfolioDecision
    assert call["output_config"] == {"effort": "medium"}
    assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert call["system"] == [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}]
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert "temperature" not in call
    assert result.transcript["request"]["output_format"] == "PortfolioDecision"
    assert result.transcript["response"]["stop_reason"] == "end_turn"


def test_parse_omits_effort_and_thinking_when_unset() -> None:
    model = _model().model_copy(update={"effort": None, "summarized_thinking": False})
    client, calls = _scripted([_Parsed(None, "end_turn", _usage())])
    result = _run_parse(client, model=model)
    assert "output_config" not in calls[0]
    assert "thinking" not in calls[0]
    assert result.output is None
    assert result.error == "The model returned no parsed output."


def test_parse_returns_refusal() -> None:
    client, calls = _scripted([_Parsed(object(), "refusal", _usage(output_tokens=20))])
    result = _run_parse(client)
    assert result.output is None
    assert result.error == "The model refused the request."
    assert result.usage.output_tokens == 20
    assert len(calls) == 1


def test_parse_returns_max_tokens() -> None:
    client, calls = _scripted([_Parsed(object(), "max_tokens", _usage())])
    result = _run_parse(client)
    assert result.output is None
    assert result.error == "The model hit the max token limit."
    assert len(calls) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 529])
def test_parse_retries_retryable_statuses(status: int) -> None:
    client, calls = _scripted([_api_error(status, "busy"), _Parsed("ok", "end_turn", _usage())])
    result = _run_parse(client, output_type=str)
    assert result.output == "ok"
    assert result.error is None
    assert len(calls) == 2


def test_parse_retries_validation_once() -> None:
    client, calls = _scripted([_validation_error(), _Parsed("ok", "end_turn", _usage())])
    result = _run_parse(client, output_type=str)
    assert result.output == "ok"
    assert len(calls) == 2


def test_parse_returns_validation_error_after_retry() -> None:
    client, calls = _scripted([_validation_error(), _validation_error()])
    result = _run_parse(client)
    assert result.output is None
    assert result.error is not None
    assert "validation" in result.error
    assert len(calls) == 2


def test_parse_returns_client_error_without_retry() -> None:
    client, calls = _scripted([_api_error(400, "bad thinking")])
    result = _run_parse(client)
    assert result.output is None
    assert result.error == "Claude API returned status 400: bad thinking"
    assert len(calls) == 1


def test_parse_returns_error_when_the_retry_also_fails() -> None:
    client, calls = _scripted([_api_error(500, "busy"), _api_error(500, "still busy")])
    result = _run_parse(client)
    assert result.error == "Claude API returned status 500: still busy"
    assert len(calls) == 2


def test_parse_returns_connection_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client, calls = _scripted([APIConnectionError(request=request)])
    result = _run_parse(client)
    assert result.error == "Claude API request failed: Connection error."
    assert len(calls) == 1


def test_parse_lets_programming_errors_raise() -> None:
    client, _calls = _scripted([TypeError("bug")])
    with pytest.raises(TypeError, match="bug"):
        _run_parse(client)


def test_client_uses_timeout_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeAsync:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr("fund.brain.llm.AsyncAnthropic", FakeAsync)
    Claude()
    assert seen["timeout"] == 300
    assert seen["max_retries"] == 2


def test_search_reads_text_blocks_by_type() -> None:
    hidden = _Block("thinking", text="secret thoughts")
    cited = _Block(
        "text",
        text="Macro fact.",
        citations=[{"type": "web_search_result_location", "url": "https://news.example/story"}],
    )
    message = _SearchMessage("end_turn", [hidden, _Block("server_tool_use"), cited], _usage(searches=2))
    client, calls = _scripted([message])
    digest, usage, _transcript = asyncio.run(
        compile_news(Claude(client), _model(), _news(), load_config().universe, NOW)
    )
    assert digest is not None
    assert digest.text == "Macro fact."
    assert "secret thoughts" not in digest.text
    assert digest.sources == ["https://news.example/story"]
    assert digest.compiled_at == NOW
    assert usage is not None
    assert usage.web_searches == 2
    assert calls[0]["tools"] == [{"type": "web_search_20260318", "name": "web_search", "max_uses": 6}]
    assert calls[0]["messages"][0]["content"].startswith("Current time: 2026-10-19T12:00:00Z. Markets: ")
    assert "S&P 500 (xyz:SP500)" in calls[0]["messages"][0]["content"]
    assert "Ether (ETH)" in calls[0]["messages"][0]["content"]


def test_search_continues_a_paused_turn_at_most_three_times() -> None:
    pauses = [
        _SearchMessage(
            "pause_turn",
            [_Block("thinking", "hidden"), _Block("text", "A")],
            _usage(input_tokens=1000, searches=1),
        )
        for _ in range(4)
    ]
    client, calls = _scripted(pauses)
    result = _run_search(client)
    assert len(calls) == 4
    assert result.output == "AAAA"
    assert result.error is None
    assert calls[1]["messages"][1]["role"] == "assistant"
    assert result.usage.input_tokens == 4000
    assert result.usage.web_searches == 4
    assert result.usage.cost_usd == pytest.approx(4000 * 4 / 1_000_000 + 4 * 0.01)


def test_search_joins_a_paused_turn_then_stops() -> None:
    first = _SearchMessage(
        "pause_turn",
        [_Block("thinking", "hidden"), _Block("text", "Part ")],
        _usage(searches=1),
    )
    second = _SearchMessage("end_turn", [_Block("text", "done.")], _usage(searches=2))
    client, calls = _scripted([first, second])
    result = _run_search(client)
    assert result.output == "Part done."
    assert "hidden" not in (result.output or "")
    assert len(calls) == 2
    assert result.usage.web_searches == 3


def test_search_falls_back_when_the_tool_version_is_rejected() -> None:
    done = _SearchMessage("end_turn", [_Block("text", "Digest.")], _usage())
    client, calls = _scripted([_api_error(400, "unknown tool"), done])
    result = _run_search(client)
    assert result.output == "Digest."
    assert calls[0]["tools"][0]["type"] == "web_search_20260318"
    assert calls[1]["tools"][0]["type"] == "web_search_20250305"


def test_search_does_not_fall_back_from_the_older_tool() -> None:
    settings = _news().model_copy(update={"tool_type": "web_search_20250305"})
    client, calls = _scripted([_api_error(400, "nope")])
    result = _run_search(client, settings)
    assert result.output is None
    assert result.error is not None
    assert "400" in result.error
    assert len(calls) == 1


def test_search_does_not_retry_other_client_errors() -> None:
    client, calls = _scripted([_api_error(401, "unauthorized")])
    result = _run_search(client)
    assert len(calls) == 1
    assert result.error is not None
    assert "401" in result.error


def test_search_retries_server_errors() -> None:
    done = _SearchMessage("end_turn", [_Block("text", "After retry.")], _usage(input_tokens=10))
    client, calls = _scripted([_api_error(500, "down"), done])
    result = _run_search(client)
    assert result.output == "After retry."
    assert len(calls) == 2
    assert calls[0]["tools"][0]["type"] == calls[1]["tools"][0]["type"] == "web_search_20260318"
    assert result.usage.input_tokens == 10


def test_search_retries_the_fallback_after_a_server_error() -> None:
    done = _SearchMessage("end_turn", [_Block("text", "Recovered.")], _usage(searches=None))
    client, calls = _scripted([_api_error(400, "unknown tool"), _api_error(529, "overloaded"), done])
    result = _run_search(client)
    assert result.output == "Recovered."
    assert [call["tools"][0]["type"] for call in calls] == [
        "web_search_20260318",
        "web_search_20250305",
        "web_search_20250305",
    ]
    assert result.usage.web_searches == 0


def test_search_returns_connection_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client, calls = _scripted([APIConnectionError(request=request)])
    result = _run_search(client)
    assert len(calls) == 1
    assert result.error == "Claude API request failed: Connection error."


def test_search_returns_max_tokens_and_empty_text() -> None:
    limited = _SearchMessage("max_tokens", [_Block("text", "partial")], _usage())
    client, _calls = _scripted([limited])
    limited_result = _run_search(client)
    assert limited_result.output is None
    assert limited_result.error == "The model hit the max token limit during the news search."
    empty = _SearchMessage("end_turn", [_Block("thinking", "only")], _usage())
    client, _calls = _scripted([empty])
    empty_result = _run_search(client)
    assert empty_result.error == "The model returned no digest text."


def test_search_returns_refusal() -> None:
    message = _SearchMessage("refusal", [_Block("text", "no")], _usage())
    client, _calls = _scripted([message])
    result = _run_search(client)
    assert result.output is None
    assert result.error == "The model refused the news search."


def test_news_failure_returns_no_digest() -> None:
    class Stub:
        def __init__(self) -> None:
            self.called = False

        async def parse(self, **kwargs: Any) -> LlmResult[Any]:
            raise AssertionError(kwargs)

        async def search(self, **kwargs: Any) -> LlmResult[str]:
            self.called = True
            return LlmResult(
                None,
                _zero("news", _model()),
                {"request": {}, "response": None},
                "search failed",
            )

    digest, usage, transcript = _run_news(Stub())
    assert digest is None
    assert usage is not None
    assert transcript is not None


def test_news_disabled_skips_the_search() -> None:
    class Stub:
        async def parse(self, **kwargs: Any) -> LlmResult[Any]:
            raise AssertionError(kwargs)

        async def search(self, **kwargs: Any) -> LlmResult[str]:
            raise AssertionError(kwargs)

    settings = _news().model_copy(update={"enabled": False})
    assert _run_news(Stub(), settings=settings) == (None, None, None)


def test_news_collects_citation_urls_from_every_turn() -> None:
    transcript = {
        "responses": [
            {
                "content": [
                    {"type": "thinking", "thinking": "secret"},
                    {
                        "type": "text",
                        "text": "Macro\nFact.",
                        "citations": [
                            {"type": "web_search_result_location", "url": "https://a.example/1"},
                            {"type": "web_search_result_location", "url": "https://a.example/1"},
                            {"type": "char_location"},
                            "nope",
                        ],
                    },
                ]
            },
            {
                "content": [
                    {
                        "type": "text",
                        "text": "More",
                        "citations": [{"type": "web_search_result_location", "url": "https://b.example/2"}],
                    }
                ]
            },
        ],
        "response": {
            "content": [
                {
                    "type": "text",
                    "text": "ignored",
                    "citations": [{"type": "web_search_result_location", "url": "https://ignored.example"}],
                }
            ]
        },
    }

    class Stub:
        async def parse(self, **kwargs: Any) -> LlmResult[Any]:
            raise AssertionError(kwargs)

        async def search(self, **kwargs: Any) -> LlmResult[str]:
            return LlmResult("Macro\nFact.\n", _zero("news", _model()), transcript, None)

    digest, _usage, _transcript = _run_news(Stub())
    assert digest is not None
    assert digest.sources == ["https://a.example/1", "https://b.example/2"]


def test_news_rejects_a_naive_clock() -> None:
    class Stub:
        async def parse(self, **kwargs: Any) -> LlmResult[Any]:
            raise AssertionError(kwargs)

        async def search(self, **kwargs: Any) -> LlmResult[str]:
            raise AssertionError(kwargs)

    naive = datetime(2026, 10, 19, 12, 0)  # noqa: DTZ001  # this test rejects a naive datetime
    with pytest.raises(ValueError, match="timezone aware"):
        _run_news(Stub(), universe=[], now=naive)


def test_news_without_citations_has_no_sources() -> None:
    class Stub:
        async def parse(self, **kwargs: Any) -> LlmResult[Any]:
            raise AssertionError(kwargs)

        async def search(self, **kwargs: Any) -> LlmResult[str]:
            return LlmResult("Fact.", _zero("news", _model()), {"response": None}, None)

    digest, _usage, _transcript = _run_news(Stub(), universe=[])
    assert digest is not None
    assert digest.sources == []


def test_empty_digest_counts_as_failure() -> None:
    class Stub:
        async def parse(self, **kwargs: Any) -> LlmResult[Any]:
            raise AssertionError(kwargs)

        async def search(self, **kwargs: Any) -> LlmResult[str]:
            return LlmResult("  \n", _zero("news", _model()), {"response": None}, None)

    digest, usage, _transcript = _run_news(Stub(), universe=[])
    assert digest is None
    assert usage is not None


def test_single_decision_sends_the_snapshot() -> None:
    recorder = Recording()
    policy = make_policy()
    result = asyncio.run(decide(_single(), recorder, make_snapshot(), make_portfolio(), _track(), policy))
    assert result.decision is not None
    assert result.error is None
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["output_type"] is PortfolioDecision
    assert call["system"] == SYSTEM_PM.format(limits=render_limits(policy))
    assert call["user"].startswith("CYCLE ")
    assert "xyz:SP500" in call["user"]
    assert len(result.usage) == 1
    assert len(result.transcripts) == 1


def test_single_decision_keeps_the_error() -> None:
    recorder = Recording(error="model broke")
    result = asyncio.run(
        decide(_single(), recorder, make_snapshot(), make_portfolio(), _track(), make_policy())
    )
    assert result.decision is None
    assert result.error == "model broke"
    assert len(result.usage) == 1


def test_team_runs_the_manager_when_one_analyst_fails() -> None:
    recorder = Recording(fail_groups={"crypto"})
    policy = make_policy()
    result = asyncio.run(decide(_team(), recorder, make_snapshot(), make_portfolio(), _track(), policy))
    assert result.decision is not None
    assert result.error == "crypto analyst failed: analyst broke"
    assert [call["purpose"] for call in recorder.calls] == [
        "analyst:opus-5.5/team:equities",
        "analyst:opus-5.5/team:macro",
        "analyst:opus-5.5/team:crypto",
        "decision:opus-5.5/team",
    ]
    equities, macro, crypto, manager = recorder.calls
    assert equities["system"] == SYSTEM_ANALYST.format(
        group="equities",
        markets=_listed(equities["system"]),
    )
    assert "S&P 500 (xyz:SP500)" in equities["system"]
    assert "xyz:SP500" in equities["user"]
    assert "BTC" not in equities["user"]
    assert "xyz:GOLD" in macro["user"]
    assert "BTC" not in macro["user"]
    assert "BTC" in crypto["user"]
    assert "xyz:SP500" not in crypto["user"]
    assert manager["system"] == SYSTEM_TEAM_PM.format(limits=render_limits(policy))
    assert "equities summary" in manager["user"]
    assert "macro summary" in manager["user"]
    assert "crypto summary" not in manager["user"]
    assert len(result.usage) == 4
    assert len(result.transcripts) == 4


def _listed(system: str) -> str:
    start = system.index("You cover these markets: ") + len("You cover these markets: ")
    end = system.index(". You receive")
    return system[start:end]


def test_team_notes_a_manager_failure() -> None:
    recorder = Recording(error="manager broke")
    result = asyncio.run(
        decide(_team(), recorder, make_snapshot(), make_portfolio(), _track(), make_policy())
    )
    assert result.decision is None
    assert result.error == "manager broke"


def test_team_skips_groups_with_no_markets() -> None:
    snapshot = make_snapshot(markets=[make_features("BTC"), make_features("ETH")])
    recorder = Recording()
    result = asyncio.run(decide(_team(), recorder, snapshot, make_portfolio(), _track(), make_policy()))
    assert result.decision is not None
    assert [call["purpose"] for call in recorder.calls] == [
        "analyst:opus-5.5/team:crypto",
        "decision:opus-5.5/team",
    ]


def test_fake_decision_follows_the_gap_rule() -> None:
    snapshot = make_snapshot(
        markets=[
            make_features("xyz:SP500", sma20_gap=0.2),
            make_features("xyz:GOLD", sma20_gap=-0.2),
            make_features("BTC", sma20_gap=0.05),
            make_features("ETH", sma20_gap=0.0),
        ]
    )
    fake = FakeClaude()
    fake.prime(snapshot)
    result = asyncio.run(
        fake.parse(
            purpose="decision",
            model=_model(),
            system="s",
            user="ignore this text",
            output_type=PortfolioDecision,
        )
    )
    assert result.error is None
    assert result.usage.cost_usd == 0.0
    decision = result.output
    assert decision is not None
    by_symbol = {item.symbol: item for item in decision.decisions}
    assert by_symbol["xyz:SP500"].prob_up_24h == pytest.approx(0.65)
    assert by_symbol["xyz:SP500"].target_weight == pytest.approx(0.1)
    assert by_symbol["xyz:GOLD"].prob_up_24h == pytest.approx(0.35)
    assert by_symbol["xyz:GOLD"].target_weight == pytest.approx(-0.1)
    assert by_symbol["BTC"].prob_up_24h == pytest.approx(0.6)
    assert by_symbol["BTC"].target_weight == pytest.approx(0.08)
    assert by_symbol["ETH"].prob_up_24h == pytest.approx(0.5)
    assert by_symbol["ETH"].target_weight == pytest.approx(0.0)
    assert all(item.stop_loss_pct == 5.0 for item in decision.decisions)


def test_fake_analyst_is_neutral() -> None:
    snapshot = make_snapshot(markets=[make_features("BTC"), make_features("ETH"), make_features("xyz:GOLD")])
    fake = FakeClaude()
    fake.prime(snapshot)
    result = asyncio.run(
        fake.parse(
            purpose="analyst",
            model=_model(),
            system="s",
            user="Look at BTC only",
            output_type=AnalystReport,
        )
    )
    report = result.output
    assert report is not None
    assert report.summary == "No directional view for this group."
    assert [view.symbol for view in report.views] == ["BTC"]
    assert report.views[0].stance == "neutral"
    assert report.views[0].prob_up_24h == 0.5


def test_fake_search_is_free() -> None:
    digest, usage, _transcript = asyncio.run(
        compile_news(FakeClaude(), _model(), _news(), load_config().universe, NOW)
    )
    assert digest is not None
    assert "Nothing notable." in digest.text
    assert digest.sources == ["https://example.com/digest"]
    assert usage is not None
    assert usage.cost_usd == 0.0
    assert usage.web_searches == 0
    assert usage.input_tokens == 0


def test_fake_team_decision_has_zero_cost() -> None:
    snapshot = make_snapshot()
    fake = FakeClaude()
    fake.prime(snapshot)
    result = asyncio.run(decide(_team(), fake, snapshot, make_portfolio(), _track(), make_policy()))
    assert result.error is None
    assert result.decision is not None
    assert len(result.usage) == 4
    assert all(item.cost_usd == 0.0 for item in result.usage)


def test_fake_requires_a_primed_snapshot() -> None:
    fake = FakeClaude()
    with pytest.raises(ValueError, match="prime"):
        asyncio.run(
            fake.parse(
                purpose="decision",
                model=_model(),
                system="s",
                user="u",
                output_type=PortfolioDecision,
            )
        )


def test_fake_rejects_an_unknown_output_type() -> None:
    class Other(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str

    fake = FakeClaude()
    fake.prime(make_snapshot())
    with pytest.raises(TypeError, match="Other"):
        asyncio.run(fake.parse(purpose="decision", model=_model(), system="s", user="u", output_type=Other))


class _Ping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


@pytest.mark.live
@pytest.mark.parametrize("model_key", ["opus-5.5", "sonnet-5.5", "haiku-5.5"])
def test_live_model_parses_a_small_answer(model_key: str) -> None:
    model = load_config().catalog[model_key].model_copy(update={"max_tokens": 256})
    result = asyncio.run(
        Claude().parse(
            purpose="live",
            model=model,
            system="Reply with the structured answer.",
            user="Say hello.",
            output_type=_Ping,
        )
    )
    assert result.error is None
    assert result.output is not None
    assert result.output.answer
