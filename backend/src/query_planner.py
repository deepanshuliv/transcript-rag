"""DeepSeek query understanding with strict local Pydantic validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import ValidationError

from .llm import OpenRouterClient
from .schemas import QueryPlan


QUERY_PLANNER_SYSTEM_PROMPT = """Return only one JSON object matching the QueryPlan schema.
Do not add fields.
Do not remove fields.
Do not return Markdown, explanations, or code fences.
Keep original_query exactly equal to the user's query.
Create one to five search_queries.
Use intent only from the allowed enum values.
"""

QUERY_PLAN_SCHEMA_NAME = "query_plan"


class QueryPlanClient(Protocol):
    """Minimal client interface needed by ``build_query_plan``."""

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        json_schema: dict[str, Any],
        schema_name: str,
    ) -> str: ...


class QueryPlanValidationError(ValueError):
    """Raised when model output is not an acceptable QueryPlan."""


def _validate_query_plan(raw_json: object, user_query: str) -> QueryPlan:
    """Parse and validate model JSON, including exact query preservation."""

    if not isinstance(raw_json, str):
        raise QueryPlanValidationError("model response content must be a JSON string")
    try:
        plan = QueryPlan.model_validate_json(raw_json)
    except ValidationError as exc:
        raise QueryPlanValidationError(str(exc)) from exc
    if plan.original_query != user_query:
        raise QueryPlanValidationError(
            "original_query must exactly equal the user's query"
        )
    return plan


def _fallback_query_plan(user_query: str) -> QueryPlan:
    """Create a retrieval-safe plan without relying on model interpretation."""

    return QueryPlan(
        original_query=user_query,
        normalized_query=user_query.strip() or user_query,
        search_queries=[user_query],
        countries=[],
        experts=[],
        intent="unknown",
        requires_all_countries=False,
    )


def _retry_messages(
    base_messages: list[dict[str, str]],
    raw_json: object,
    validation_error: QueryPlanValidationError,
) -> list[dict[str, str]]:
    """Give the model one correction attempt with the local validation error."""

    messages = list(base_messages)
    if isinstance(raw_json, str):
        messages.append({"role": "assistant", "content": raw_json})
    messages.append(
        {
            "role": "user",
            "content": (
                "Your previous response failed local QueryPlan validation. "
                f"Validation error: {validation_error}\n"
                "Return one corrected JSON object only, matching the supplied "
                "QueryPlan schema. Keep original_query exactly equal to the "
                "user's query."
            ),
        }
    )
    return messages


def build_query_plan(
    user_query: str,
    *,
    client: QueryPlanClient | None = None,
) -> QueryPlan:
    """Build a validated DeepSeek query plan with one retry and safe fallback.

    The original query is never replaced by model output. Retrieval callers
    receive only a Pydantic-validated plan, including the deterministic
    fallback when both model responses fail validation.
    """

    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be a non-empty string")

    planner_client = client or OpenRouterClient()
    schema = QueryPlan.model_json_schema()
    messages = [
        {"role": "system", "content": QUERY_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    raw_json = planner_client.complete_json(
        messages,
        json_schema=schema,
        schema_name=QUERY_PLAN_SCHEMA_NAME,
    )
    try:
        return _validate_query_plan(raw_json, user_query)
    except QueryPlanValidationError as first_error:
        retry_raw_json = planner_client.complete_json(
            _retry_messages(messages, raw_json, first_error),
            json_schema=schema,
            schema_name=QUERY_PLAN_SCHEMA_NAME,
        )
        try:
            return _validate_query_plan(retry_raw_json, user_query)
        except QueryPlanValidationError:
            return _fallback_query_plan(user_query)
