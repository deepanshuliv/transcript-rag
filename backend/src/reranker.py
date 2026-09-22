"""OpenRouter Cohere reranking with a deterministic no-op fallback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from .config import settings
from .schemas import EvidenceItem


class Reranker(Protocol):
    """Provider-independent reranking interface."""

    def rerank(self, query: str, items: list[EvidenceItem]) -> list[EvidenceItem]: ...


class NoOpReranker:
    """Keep RRF order when no reranking provider is available."""

    def rerank(self, query: str, items: list[EvidenceItem]) -> list[EvidenceItem]:
        return list(items)


class OpenRouterCohereReranker:
    """Call OpenRouter's Cohere-compatible ``/api/v1/rerank`` endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        http_client: Any | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.model = model or settings.rerank_model
        self.endpoint = self.base_url.rstrip("/") + "/rerank"
        self.http_client = http_client or httpx.Client(timeout=timeout)
        self.last_error: Exception | None = None

    def rerank(self, query: str, items: list[EvidenceItem]) -> list[EvidenceItem]:
        """Rerank candidates while preserving every original evidence record."""

        original_items = list(items)
        if len(original_items) <= 1 or not self.api_key:
            return original_items

        try:
            response = self.http_client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [item.text for item in original_items],
                    "top_n": len(original_items),
                },
            )
            response.raise_for_status()
            payload = response.json()
            raw_results = payload["results"]
            if not isinstance(raw_results, list):
                raise ValueError("reranker response results must be a list")

            reranked: list[EvidenceItem] = []
            used_indices: set[int] = set()
            for raw_result in raw_results:
                if not isinstance(raw_result, dict):
                    raise ValueError("reranker result must be an object")
                index = raw_result["index"]
                if not isinstance(index, int) or not 0 <= index < len(original_items):
                    raise ValueError("reranker result index is out of range")
                if index in used_indices:
                    continue
                score_value = raw_result.get(
                    "relevance_score", raw_result.get("score")
                )
                if score_value is None:
                    raise ValueError("reranker result has no relevance score")
                reranked.append(
                    original_items[index].model_copy(
                        update={"rerank_score": float(score_value)}
                    )
                )
                used_indices.add(index)

            # Providers may omit candidates; retain them after returned results
            # instead of silently losing evidence.
            reranked.extend(
                item
                for index, item in enumerate(original_items)
                if index not in used_indices
            )
            self.last_error = None
            return reranked
        except Exception as exc:  # provider/rate-limit/malformed-response fallback
            self.last_error = exc
            return original_items
