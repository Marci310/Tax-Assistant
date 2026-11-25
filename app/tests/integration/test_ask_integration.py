import os

import httpx
import pytest

pytestmark = pytest.mark.integration

ASK = "/api/v1/ask"


@pytest.fixture
def base_url() -> str:
    return os.environ.get("BASE_URL", "http://localhost:8000")


@pytest.fixture
async def client(base_url):
    async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
        try:
            await client.get("/health")
        except httpx.TransportError:
            pytest.fail(f"Stack not reachable at {base_url}, start it with: docker compose --profile local up -d")
        yield client


def assert_has_real_sources(data: dict) -> None:
    assert len(data["sources"]) > 0
    for source in data["sources"]:
        assert source["name"]
        assert source["page_number"] >= 1
        assert source["similarity"] >= 0.45


async def test_health(client):
    response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["indexed_chunks"] > 0


async def test_ask_hungarian(client):
    response = await client.post(ASK, json={"question": "Mi a növekedési adóhitel?", "top_k": 4})

    assert response.status_code == 200
    data = response.json()
    assert data["answered"] is True
    assert "adóhitel" in data["answer"].lower()
    assert_has_real_sources(data)


async def test_ask_english(client):
    response = await client.post(ASK, json={"question": "What is the corporate tax rate?", "top_k": 4})

    assert response.status_code == 200
    data = response.json()
    assert data["answered"] is True
    assert_has_real_sources(data)


async def test_off_topic(client):
    response = await client.post(ASK, json={"question": "Hogyan kell pizzát sütni?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answered"] is False
    assert data["sources"] == []


async def test_invalid_input(client):
    response = await client.post(ASK, json={"question": "Hi", "top_k": 4})

    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "invalid_request"
    assert data["message"].startswith("question:")
