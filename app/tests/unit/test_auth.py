from unittest.mock import ANY, AsyncMock

import pytest
from pydantic import SecretStr

from core.config import Settings

ASK = "/api/v1/ask"
QUESTION = {"question": "Mi az elismert költség?"}
API_KEY = "test-api-key"


@pytest.fixture
def secured_settings(settings) -> Settings:
    return settings.model_copy(update={"api_key": SecretStr(API_KEY)})


@pytest.fixture
def answering_graph() -> AsyncMock:
    graph = AsyncMock()
    graph.ainvoke.return_value = {"answer": "Válasz", "answered": True, "documents": []}
    return graph


@pytest.mark.parametrize("headers", [
    {},
    {"X-API-Key": "wrong-key"},
    {"X-API-Key": ""},
    {"X-API-Key": "é".encode("latin-1")},
])
def test_missing_or_wrong_key_returns_401(headers, secured_settings, answering_graph, make_client):
    client = make_client(answering_graph, secured_settings)

    response = client.post(ASK, json=QUESTION, headers=headers)

    assert response.status_code == 401
    assert response.json() == {
        "error": "unauthorized",
        "message": "Missing or invalid API key.",
        "request_id": ANY,
        "timestamp": ANY,
    }
    answering_graph.ainvoke.assert_not_awaited()


def test_correct_key_is_accepted(secured_settings, answering_graph, make_client):
    client = make_client(answering_graph, secured_settings)

    response = client.post(ASK, json=QUESTION, headers={"X-API-Key": API_KEY})

    assert response.status_code == 200
    answering_graph.ainvoke.assert_awaited_once()


def test_auth_is_off_without_a_configured_key(answering_graph, make_client):
    client = make_client(answering_graph)

    response = client.post(ASK, json=QUESTION)

    assert response.status_code == 200


def test_unauthenticated_request_is_rejected_before_validation(secured_settings, answering_graph, make_client):
    client = make_client(answering_graph, secured_settings)

    response = client.post(ASK, json={"question": ""})

    assert response.status_code == 401


def test_health_needs_no_key(secured_settings, qdrant_mock, make_client):
    client = make_client(AsyncMock(), secured_settings)

    response = client.get("/health")

    assert response.status_code == 200


@pytest.mark.parametrize("value", ["", None])
def test_empty_api_key_means_auth_off(value):
    settings = Settings(_env_file=None, llm_provider="ollama", api_key=value)

    assert settings.api_key is None
