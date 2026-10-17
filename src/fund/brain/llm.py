# pyright: strict
"""Claude client. Model and API failures return an error instead of raising."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, cast

from anthropic import APIError, APIStatusError, AsyncAnthropic
from anthropic.types import (
    Message,
    OutputConfigParam,
    ParsedMessage,
    TextBlockParam,
    ThinkingConfigAdaptiveParam,
    Usage,
)
from pydantic import ValidationError

from fund.brain.costs import usage_cost
from fund.config import ModelSpec, NewsSettings
from fund.contracts import LlmUsage

logger = logging.getLogger(__name__)

_FALLBACK_TOOL = "web_search_20250305"
_MAX_CONTINUATIONS = 3


@dataclass(frozen=True)
class LlmResult[T]:
    """One Claude call: parsed output when it worked, plus usage and the transcript."""

    output: T | None
    usage: LlmUsage
    transcript: dict[str, Any]
    error: str | None


class Llm(Protocol):
    """parse and search, implemented by Claude and by FakeClaude."""

    async def parse[T](
        self,
        *,
        purpose: str,
        model: ModelSpec,
        system: str,
        user: str,
        output_type: type[T],
    ) -> LlmResult[T]: ...

    async def search(
        self,
        *,
        purpose: str,
        model: ModelSpec,
        system: str,
        user: str,
        news: NewsSettings,
    ) -> LlmResult[str]: ...


@dataclass
class _Tokens:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    searches: int = 0

    def add_usage(self, usage: Usage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_creation_input_tokens += usage.cache_creation_input_tokens or 0
        self.cache_read_input_tokens += usage.cache_read_input_tokens or 0
        server = usage.server_tool_use
        if server is not None:
            self.searches += server.web_search_requests

    def add(self, other: _Tokens) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.searches += other.searches


@dataclass
class _SearchRun:
    text: str
    error: str | None
    status: int | None
    retryable: bool
    tokens: _Tokens
    request: dict[str, Any]
    responses: list[dict[str, Any]]


class _Retry(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _GiveUp(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class Claude:
    """Async Claude client with one application retry and no raised model failures."""

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        if client is None:
            client = AsyncAnthropic(timeout=300, max_retries=2)
        self._client = client

    async def parse[T](
        self,
        *,
        purpose: str,
        model: ModelSpec,
        system: str,
        user: str,
        output_type: type[T],
    ) -> LlmResult[T]:
        """Ask Claude for one structured object. Refusal, truncation, and API errors are returned."""
        kwargs = _parse_kwargs(model, system, user, output_type)
        try:
            try:
                response = await self._parse_once(kwargs)
            except _Retry:
                response = await self._parse_once(kwargs)
        except (_Retry, _GiveUp) as exc:
            return cast(LlmResult[T], _failure(purpose, model, kwargs, exc.message))
        return _from_parse(purpose, model, kwargs, response)

    async def search(
        self,
        *,
        purpose: str,
        model: ModelSpec,
        system: str,
        user: str,
        news: NewsSettings,
    ) -> LlmResult[str]:
        """Run a web search and concatenate the text blocks. A rejected tool version falls back once."""
        runs = [await self._search_once(model, system, user, news, news.tool_type)]
        if runs[-1].retryable:
            runs.append(await self._search_once(model, system, user, news, news.tool_type))
        elif _rejected_tool(runs[-1]) and news.tool_type != _FALLBACK_TOOL:
            runs.append(await self._search_once(model, system, user, news, _FALLBACK_TOOL))
            if runs[-1].retryable:
                runs.append(await self._search_once(model, system, user, news, _FALLBACK_TOOL))
        result = _search_result(purpose, model, news.search_cost_usd, runs)
        if result.error:
            logger.warning("Claude call %s failed: %s", purpose, result.error)
        return result

    async def _parse_once(self, kwargs: dict[str, Any]) -> ParsedMessage[Any]:
        try:
            return await self._client.messages.parse(**kwargs)
        except ValidationError as exc:
            raise _Retry(f"Model output failed validation: {exc}") from exc
        except APIStatusError as exc:
            message = f"Claude API returned status {exc.status_code}: {exc.message}"
            if _retryable_status(exc.status_code):
                raise _Retry(message) from exc
            raise _GiveUp(message) from exc
        except APIError as exc:
            raise _GiveUp(f"Claude API request failed: {exc.message}") from exc

    async def _search_once(
        self,
        model: ModelSpec,
        system: str,
        user: str,
        news: NewsSettings,
        tool_type: str,
    ) -> _SearchRun:
        messages: list[Any] = [_user_message(user)]
        tools = _search_tools(tool_type, news.max_searches)
        tokens = _Tokens()
        dumped: list[dict[str, Any]] = []
        text_parts: list[str] = []
        response: Message | None = None
        try:
            for turn in range(_MAX_CONTINUATIONS + 1):
                payload = _create_kwargs(model, system, list(messages), tools)
                response = cast(Message, await self._client.messages.create(**payload))
                tokens.add_usage(response.usage)
                dumped.append(response.model_dump(mode="json"))
                text_parts.append(_text_parts(response))
                if response.stop_reason != "pause_turn" or turn == _MAX_CONTINUATIONS:
                    break
                messages.append(response.to_param())
        except APIStatusError as exc:
            message = f"Claude API returned status {exc.status_code}: {exc.message}"
            return _SearchRun(
                "",
                message,
                exc.status_code,
                _retryable_status(exc.status_code),
                tokens,
                _create_kwargs(model, system, list(messages), tools),
                dumped,
            )
        except APIError as exc:
            return _SearchRun(
                "",
                f"Claude API request failed: {exc.message}",
                None,
                False,
                tokens,
                _create_kwargs(model, system, list(messages), tools),
                dumped,
            )
        text = "".join(text_parts)
        stop = None if response is None else response.stop_reason
        return _SearchRun(
            text,
            _digest_error(stop, text),
            None,
            False,
            tokens,
            _create_kwargs(model, system, list(messages), tools),
            dumped,
        )


def _from_parse[T](
    purpose: str,
    model: ModelSpec,
    kwargs: dict[str, Any],
    response: ParsedMessage[T],
) -> LlmResult[T]:
    usage = _billed(purpose, model, _tokens_from(response.usage), 0.0)
    transcript = _transcript(kwargs, response.model_dump(mode="json"))
    if response.stop_reason == "refusal":
        result = _empty("The model refused the request.", usage, transcript)
    elif response.stop_reason == "max_tokens":
        result = _empty("The model hit the max token limit.", usage, transcript)
    elif response.parsed_output is None:
        result = _empty("The model returned no parsed output.", usage, transcript)
    else:
        result = LlmResult(response.parsed_output, usage, transcript, None)
    typed: LlmResult[T] = cast(LlmResult[T], result)
    if typed.error:
        logger.warning("Claude call %s failed: %s", purpose, typed.error)
    return typed


def _failure(purpose: str, model: ModelSpec, kwargs: dict[str, Any], message: str) -> LlmResult[Any]:
    logger.warning("Claude call %s failed: %s", purpose, message)
    return LlmResult(None, _billed(purpose, model, _Tokens(), 0.0), _transcript(kwargs, None), message)


def _empty(message: str, usage: LlmUsage, transcript: dict[str, Any]) -> LlmResult[Any]:
    return LlmResult(None, usage, transcript, message)


def _search_result(
    purpose: str,
    model: ModelSpec,
    search_cost: float,
    runs: list[_SearchRun],
) -> LlmResult[str]:
    last = runs[-1]
    tokens = _Tokens()
    for run in runs:
        tokens.add(run.tokens)
    transcript: dict[str, Any] = {
        "request": _public_request(last.request),
        "response": last.responses[-1] if last.responses else None,
        "responses": last.responses,
    }
    output = None if last.error else last.text
    return LlmResult(output, _billed(purpose, model, tokens, search_cost), transcript, last.error)


def _rejected_tool(run: _SearchRun) -> bool:
    return run.status == 400


def _digest_error(stop_reason: str | None, text: str) -> str | None:
    if stop_reason == "refusal":
        return "The model refused the news search."
    if stop_reason == "max_tokens":
        return "The model hit the max token limit during the news search."
    if not text:
        return "The model returned no digest text."
    return None


def _retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _parse_kwargs(model: ModelSpec, system: str, user: str, output_type: type[Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model.id,
        "max_tokens": model.max_tokens,
        "system": _system_blocks(system),
        "messages": [_user_message(user)],
        "output_format": output_type,
    }
    kwargs.update(_model_options(model))
    return kwargs


def _create_kwargs(model: ModelSpec, system: str, messages: list[Any], tools: list[Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model.id,
        "max_tokens": model.max_tokens,
        "system": _system_blocks(system),
        "messages": messages,
        "tools": tools,
    }
    kwargs.update(_model_options(model))
    return kwargs


def _model_options(model: ModelSpec) -> dict[str, Any]:
    options: dict[str, Any] = {}
    effort = _effort(model)
    if effort is not None:
        options["output_config"] = effort
    thinking = _thinking(model)
    if thinking is not None:
        options["thinking"] = thinking
    return options


def _effort(model: ModelSpec) -> OutputConfigParam | None:
    if model.effort is None:
        return None
    return {"effort": model.effort}


def _thinking(model: ModelSpec) -> ThinkingConfigAdaptiveParam | None:
    if not model.summarized_thinking:
        return None
    return {"type": "adaptive", "display": "summarized"}


def _system_blocks(system: str) -> list[TextBlockParam]:
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _user_message(user: str) -> dict[str, str]:
    return {"role": "user", "content": user}


def _search_tools(tool_type: str, max_uses: int) -> list[dict[str, Any]]:
    if tool_type == _FALLBACK_TOOL:
        return [{"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses}]
    return [{"type": "web_search_20260318", "name": "web_search", "max_uses": max_uses}]


def _text_parts(response: Message) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def _tokens_from(usage: Usage) -> _Tokens:
    tokens = _Tokens()
    tokens.add_usage(usage)
    return tokens


def _billed(purpose: str, model: ModelSpec, tokens: _Tokens, search_cost: float) -> LlmUsage:
    return LlmUsage(
        purpose=purpose,
        model_id=model.id,
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        cache_write_tokens=tokens.cache_creation_input_tokens,
        cache_read_tokens=tokens.cache_read_input_tokens,
        web_searches=tokens.searches,
        cost_usd=usage_cost(tokens, model, tokens.searches, search_cost),
    )


def _transcript(request: dict[str, Any], response: dict[str, Any] | None) -> dict[str, Any]:
    return {"request": _public_request(request), "response": response}


def _public_request(payload: dict[str, Any]) -> dict[str, Any]:
    view = dict(payload)
    output_type = view.get("output_format")
    if isinstance(output_type, type):
        view["output_format"] = output_type.__name__
    return view
