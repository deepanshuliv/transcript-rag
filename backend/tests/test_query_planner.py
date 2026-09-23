"""Phase 3 QueryPlan schema, retry, and fallback tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.llm import OpenRouterClient
from src.query_planner import (
    QUERY_PLANNER_SYSTEM_PROMPT,
    build_query_plan,
    query_plan_response_schema,
)
from src.schemas import QueryPlan


USER_QUERY = "Compare funding barriers in France and Germany"


def valid_payload(query: str = USER_QUERY) -> dict[str, Any]:
    return {
        "original_query": query,
        "normalized_query": "funding barriers in France and Germany",
        "search_queries": ["funding barriers France", "funding barriers Germany"],
        "countries": ["France", "Germany"],
        "experts": [],
        "intent": "comparison",
        "requires_all_countries": True,
    }


class FakePlannerClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete_json(self, messages: list[dict[str, str]], **kwargs: object) -> object:
        self.calls.append({"messages": messages, **kwargs})
        return self.responses.pop(0)


class FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.response


class FakeStreamingCompletions:
    def __init__(self, fragments: list[str]) -> None:
        self.fragments = fragments
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return iter(
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=fragment))]
            )
            for fragment in self.fragments
        )


def test_openrouter_client_sends_fixed_json_schema_request() -> None:
    completions = FakeCompletions(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(valid_payload()))
                )
            ]
        )
    )
    fake_openai = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client = OpenRouterClient(
        api_key="test-key",
        model="openai/gpt-4.1-nano",
        client=fake_openai,
    )

    raw_json = client.complete_json(
        [
            {"role": "system", "content": QUERY_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": USER_QUERY},
        ],
        json_schema=query_plan_response_schema(),
        schema_name="query_plan",
    )

    assert json.loads(raw_json) == valid_payload()
    assert completions.kwargs is not None
    assert completions.kwargs["model"] == "openai/gpt-4.1-nano"
    assert completions.kwargs["temperature"] == 0
    response_format = completions.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "query_plan"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == list(schema["properties"])
    assert "countries" in schema["required"]
    assert "experts" in schema["required"]
    assert "requires_all_countries" in schema["required"]
    assert "minLength" not in json.dumps(schema)
    assert "minItems" not in json.dumps(schema)


def test_openrouter_client_forwards_minimal_reasoning_to_provider() -> None:
    completions = FakeCompletions(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(valid_payload()))
                )
            ]
        )
    )
    client = OpenRouterClient(
        api_key="test-key",
        model="openai/gpt-5-nano",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    client.complete_json(
        [{"role": "user", "content": USER_QUERY}],
        json_schema=query_plan_response_schema(),
        schema_name="latency_test",
        reasoning_effort="minimal",
    )

    assert completions.kwargs is not None
    assert completions.kwargs["extra_body"] == {
        "reasoning": {"effort": "minimal"}
    }


def test_openrouter_client_streams_provider_content_deltas() -> None:
    completions = FakeStreamingCompletions(['{"answer":', '"streamed"}'])
    client = OpenRouterClient(
        api_key="test-key",
        model="openai/gpt-5-nano",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    received: list[str] = []

    content = client.complete_json(
        [{"role": "user", "content": USER_QUERY}],
        json_schema=query_plan_response_schema(),
        schema_name="stream_test",
        on_content_delta=received.append,
    )

    assert completions.kwargs is not None
    assert completions.kwargs["stream"] is True
    assert content == '{"answer":"streamed"}'
    assert received == ['{"answer":', '"streamed"}']


def test_valid_plan_has_exactly_the_fixed_seven_fields() -> None:
    client = FakePlannerClient([json.dumps(valid_payload())])

    plan = build_query_plan(USER_QUERY, client=client)

    assert set(plan.model_dump()) == {
        "original_query",
        "normalized_query",
        "search_queries",
        "countries",
        "experts",
        "intent",
        "requires_all_countries",
    }
    assert plan.original_query == USER_QUERY
    assert len(client.calls) == 1


def test_extra_field_is_rejected_then_valid_retry_is_accepted() -> None:
    invalid = valid_payload()
    invalid["unexpected"] = "reject me"
    client = FakePlannerClient([json.dumps(invalid), json.dumps(valid_payload())])

    plan = build_query_plan(USER_QUERY, client=client)

    assert plan.intent == "comparison"
    assert len(client.calls) == 2
    retry_messages = client.calls[1]["messages"]
    assert "validation" in retry_messages[-1]["content"].lower()


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {key: value for key, value in valid_payload().items() if key != "intent"},
        {**valid_payload(), "intent": "unsupported_intent"},
    ],
)
def test_missing_or_invalid_fields_fall_back_after_one_retry(
    invalid_payload: dict[str, Any],
) -> None:
    client = FakePlannerClient(
        [json.dumps(invalid_payload), json.dumps(invalid_payload)]
    )

    plan = build_query_plan(USER_QUERY, client=client)

    assert plan.original_query == USER_QUERY
    assert plan.normalized_query == USER_QUERY
    assert plan.search_queries == [USER_QUERY]
    assert plan.intent == "unknown"
    assert plan.countries == []
    assert plan.experts == []
    assert plan.requires_all_countries is False
    assert len(client.calls) == 2


def test_model_cannot_change_original_query() -> None:
    wrong_original = valid_payload("a different query")
    client = FakePlannerClient([json.dumps(wrong_original), json.dumps(valid_payload())])

    plan = build_query_plan(USER_QUERY, client=client)

    assert plan.original_query == USER_QUERY
    assert len(client.calls) == 2
