"""OpenRouter-compatible chat client used by structured LLM requests."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from openai import OpenAI

from .config import settings


class OpenRouterConfigurationError(RuntimeError):
    """Raised when the OpenRouter client cannot be configured safely."""


class OpenRouterResponseError(RuntimeError):
    """Raised when OpenRouter returns no usable message content."""


_UNSUPPORTED_STRICT_SCHEMA_KEYS = {
    "default",
    "description",
    "maxItems",
    "maxLength",
    "minItems",
    "minLength",
    "title",
}


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Pydantic schema for strict OpenAI-compatible output modes."""

    normalized = deepcopy(schema)

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key in list(value):
                if key in _UNSUPPORTED_STRICT_SCHEMA_KEYS:
                    del value[key]
                else:
                    visit(value[key])
            properties = value.get("properties")
            if value.get("type") == "object" and isinstance(properties, dict):
                value["required"] = list(properties)
                value["additionalProperties"] = False
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(normalized)
    return normalized


class OpenRouterClient:
    """Small OpenAI-compatible client configured for OpenRouter.

    ``client`` is injectable so query-planning tests can verify request shape
    without making a network call or requiring an API key.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.model = model or settings.query_model
        self.timeout = timeout
        if client is not None:
            self._client = client
        else:
            if not self.api_key:
                raise OpenRouterConfigurationError(
                    "OPENROUTER_API_KEY is required for OpenRouter requests"
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        json_schema: dict[str, Any],
        schema_name: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        on_content_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Request one JSON object using OpenAI-compatible structured output."""

        request_options: dict[str, Any] = {}
        if reasoning_effort is not None:
            request_options["extra_body"] = {
                "reasoning": {"effort": reasoning_effort}
            }
        response = self._client.chat.completions.create(
            model=model or self.model,
            messages=list(messages),
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            },
            stream=on_content_delta is not None,
            **request_options,
        )
        if on_content_delta is not None:
            content_parts: list[str] = []
            for event in response:
                try:
                    content = event.choices[0].delta.content
                except (AttributeError, IndexError, TypeError):
                    continue
                if isinstance(content, str) and content:
                    content_parts.append(content)
                    on_content_delta(content)
            content = "".join(content_parts)
        else:
            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError, TypeError) as exc:
                raise OpenRouterResponseError(
                    "OpenRouter response did not contain a chat message"
                ) from exc
        if not isinstance(content, str) or not content.strip():
            raise OpenRouterResponseError(
                "OpenRouter response contained empty message content"
            )
        return content


# Explicit alias for callers that use the provider name in their imports.
OpenRouterLLM = OpenRouterClient
