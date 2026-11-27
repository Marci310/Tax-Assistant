import asyncio
import random

import httpx
from langchain_core.messages import AIMessage, BaseMessage

from main import app
from rag.graph import build_graph

ASK = "/api/v1/ask"
REQUESTS = 20


class EchoChatModel:


    def __init__(self):
        self.in_flight = 0
        self.max_in_flight = 0

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(random.uniform(0.01, 0.05))
            return AIMessage(content="\n".join(str(message.content) for message in messages))
        finally:
            self.in_flight -= 1


async def test_parallel_requests_do_not_mix(settings, documents, make_vector_store, make_client):
    model = EchoChatModel()
    graph = build_graph(make_vector_store([(documents[0], 0.72)]), model, settings)
    make_client(graph)
    questions = [f"Kérdés {i:02d}: mi az elismert költség?" for i in range(REQUESTS)]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(
            *(client.post(ASK, json={"question": question}) for question in questions)
        )

    for question, response in zip(questions, responses):
        assert response.status_code == 200
        answer = response.json()["answer"]
        assert question in answer
        others = [other for other in questions if other != question and other in answer]
        assert others == []

    request_ids = {response.headers["X-Request-ID"] for response in responses}
    assert len(request_ids) == REQUESTS

    assert model.max_in_flight == settings.max_concurrent_llm_calls
