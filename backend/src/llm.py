"""OpenRouter-compatible chat client used by the Phase 3 query planner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI

from .config import settings


class OpenRouterConfigurationError(RuntimeError):
    """Raised when the OpenRouter client cannot be configured safely."""


class OpenRouterResponseError(RuntimeError):
    """Raised when OpenRouter returns no usable message content."""


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
    ) -> str:
        """Request one JSON object using OpenAI-compatible structured output."""

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
        )
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
