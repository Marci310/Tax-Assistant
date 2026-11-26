import asyncio
from unittest.mock import ANY, AsyncMock

import httpx
import ollama
import openai
import pytest
from langchain_core.language_models import FakeListChatModel
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from main import app
from rag.graph import NO_ANSWER, build_graph

ASK = "/api/v1/ask"
QUESTION = {"question": "Mi az elismert költség?"}
OPENAI_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def failing_graph(exc: Exception) -> AsyncMock:
    graph = AsyncMock()
    graph.ainvoke.side_effect = exc
    return graph


# --- Success paths ---

def test_ask_success(settings, documents, make_vector_store, make_client):
    graph = build_graph(make_vector_store([(documents[0], 0.72345)]), FakeListChatModel(responses=["Válasz"]), settings)
    client = make_client(graph)

    response = client.post(ASK, json=QUESTION)

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Válasz",
        "answered": True,
        "model": settings.chat_model,
        "timestamp": ANY,
        "sources": [
            {
                "name": documents[0].metadata["title"],
                "page_number": documents[0].metadata["page"],
                "text": documents[0].page_content,
                "similarity": 0.723,
            }
        ],
    }


def test_ask_no_hits_is_not_an_error(settings, make_vector_store, make_client):
    graph = build_graph(make_vector_store([]), FakeListChatModel(responses=[]), settings)
    client = make_client(graph)

    response = client.post(ASK, json=QUESTION)

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is False
    assert body["answer"] == NO_ANSWER["hu"]
    assert body["sources"] == []


@pytest.mark.parametrize("payload, expected_top_k", [
    (QUESTION, 4),
    ({**QUESTION, "top_k": 7}, 7),
])
def test_top_k_default_and_override(payload, expected_top_k, make_client):
    graph = AsyncMock()
    graph.ainvoke.return_value = {"answer": "Válasz", "answered": True, "documents": []}
    client = make_client(graph)

    response = client.post(ASK, json=payload)

    assert response.status_code == 200
    graph.ainvoke.assert_awaited_once_with({"question": QUESTION["question"], "top_k": expected_top_k})


# --- Validation (422) ---

@pytest.mark.parametrize("payload, field", [
    ({}, "question"),
    ({"question": ""}, "question"),
    ({"question": "     "}, "question"),
    ({"question": "ab"}, "question"),
    ({"question": "x" * 1001}, "question"),
    ({**QUESTION, "top_k": 0}, "top_k"),
    ({**QUESTION, "top_k": 11}, "top_k"),
    ({**QUESTION, "top_k": "sok"}, "top_k"),
])
def test_invalid_input_returns_422(payload, field, make_client):
    graph = AsyncMock()
    client = make_client(graph)

    response = client.post(ASK, json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_request"
    assert body["message"].startswith(f"{field}:")
    graph.ainvoke.assert_not_awaited()


def test_invalid_json_returns_422(make_client):
    client = make_client(AsyncMock())

    response = client.post(ASK, content=b'{"question":', headers={"Content-Type": "application/json"})

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert response.json()["message"] == "The request body is not valid JSON."


# --- Error mapping (502 / 503 / 504 / 500) ---

@pytest.mark.parametrize("exc, status, error", [
    (TimeoutError(), 504, "llm_timeout"),
    (httpx.ReadTimeout("slow"), 504, "llm_timeout"),
    (openai.APITimeoutError(request=OPENAI_REQUEST), 504, "llm_timeout"),
    (httpx.ConnectError("down"), 502, "llm_unavailable"),
    (ConnectionError("down"), 502, "llm_unavailable"),
    (ollama.ResponseError("model not found"), 502, "llm_unavailable"),
    (openai.APIConnectionError(request=OPENAI_REQUEST), 502, "llm_unavailable"),
    (ResponseHandlingException(Exception("down")), 503, "vector_store_unavailable"),
    (UnexpectedResponse(500, "error", b"", httpx.Headers()), 503, "vector_store_unavailable"),
    (RuntimeError("bug"), 500, "internal_error"),
])
def test_errors_are_mapped_to_status_codes(exc, status, error, make_client):
    client = make_client(failing_graph(exc))

    response = client.post(ASK, json=QUESTION)

    assert response.status_code == status
    assert response.json() == {"error": error, "message": ANY, "request_id": ANY, "timestamp": ANY}


def test_unexpected_error_does_not_leak_details(make_client):
    client = make_client(failing_graph(RuntimeError("secret database password")))

    response = client.post(ASK, json=QUESTION)

    assert response.status_code == 500
    assert "secret" not in response.text
    assert response.json()["message"] == "An unexpected error occurred."


async def slow_graph(*args, **kwargs):
    await asyncio.sleep(1)


def test_real_timeout_returns_504(settings, make_client):
    graph = AsyncMock()
    graph.ainvoke.side_effect = slow_graph
    client = make_client(graph, settings.model_copy(update={"llm_timeout_seconds": 0.05}))

    response = client.post(ASK, json=QUESTION)

    assert response.status_code == 504
    assert response.json()["error"] == "llm_timeout"


# --- Request ID ---

@pytest.mark.parametrize("exc", [httpx.ConnectError("down"), RuntimeError("bug")])
def test_request_id_in_header_matches_body(exc, make_client):
    client = make_client(failing_graph(exc))

    response = client.post(ASK, json=QUESTION)

    request_id = response.headers["X-Request-ID"]
    assert request_id
    assert response.json()["request_id"] == request_id


def test_client_request_id_is_reused(make_client):
    client = make_client(failing_graph(httpx.ConnectError("down")))
    own_id = "0123456789abcdef0123456789abcdef"

    response = client.post(ASK, json=QUESTION, headers={"X-Request-ID": own_id})

    assert response.headers["X-Request-ID"] == own_id
    assert response.json()["request_id"] == own_id


# --- Health ---

def test_health_ok(qdrant_mock, make_client):
    response = make_client(AsyncMock()).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "qdrant_healthy": True,
        "indexed_chunks": 317,
        "provider": "ollama",
        "timestamp": ANY,
    }


def test_health_degraded_when_qdrant_down(qdrant_mock, make_client):
    qdrant_mock.get_collections.side_effect = Exception("connection refused")

    response = make_client(AsyncMock()).get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["qdrant_healthy"] is False
    assert body["indexed_chunks"] == 0
    qdrant_mock.count.assert_not_called()


# --- HTTP errors outside validation (400 / 404 / 405) ---

@pytest.mark.parametrize("method, url, kwargs, status, error", [
    ("get", "/api/v1/unknown", {}, 404, "not_found"),
    ("get", ASK, {}, 405, "method_not_allowed"),
    ("post", ASK, {"content": "é".encode("latin-1"), "headers": {"Content-Type": "application/json"}}, 400, "bad_request"),
])
def test_http_errors_use_error_format(method, url, kwargs, status, error, make_client):
    client = make_client(AsyncMock())

    response = getattr(client, method)(url, **kwargs)

    assert response.status_code == status
    assert response.json() == {"error": error, "message": ANY, "request_id": ANY, "timestamp": ANY}


def test_method_not_allowed_keeps_allow_header(make_client):
    response = make_client(AsyncMock()).get(ASK)

    assert response.headers["Allow"] == "POST"
