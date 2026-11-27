import pytest
from langchain_core.language_models import FakeListChatModel
from prometheus_client.metrics import MetricWrapperBase

from core.metrics import LLM_DURATION, QUESTIONS
from rag.graph import build_graph

ASK = "/api/v1/ask"
QUESTION = {"question": "Mi az elismert költség?"}


def sample_value(metric: MetricWrapperBase, name: str, labels: dict[str, str] | None = None) -> float:
    for family in metric.collect():
        for sample in family.samples:
            if sample.name == name and sample.labels == (labels or {}):
                return sample.value
    return 0.0


def questions_count(answered: str) -> float:
    return sample_value(QUESTIONS, "tax_assistant_questions_total", {"answered": answered})


@pytest.mark.parametrize("score, answered", [(0.72, "true"), (0.30, "false")])
def test_question_outcome_is_counted(score, answered, settings, documents, make_vector_store, make_client):
    graph = build_graph(make_vector_store([(documents[0], score)]), FakeListChatModel(responses=["Válasz"]), settings)
    client = make_client(graph)
    opposite = "false" if answered == "true" else "true"
    before, opposite_before = questions_count(answered), questions_count(opposite)

    response = client.post(ASK, json=QUESTION)

    assert response.status_code == 200
    assert questions_count(answered) == before + 1
    assert questions_count(opposite) == opposite_before


def test_metrics_endpoint_exposes_the_counter(make_client, settings, documents, make_vector_store):
    graph = build_graph(make_vector_store([(documents[0], 0.72)]), FakeListChatModel(responses=["Válasz"]), settings)
    client = make_client(graph)
    client.post(ASK, json=QUESTION)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert 'tax_assistant_questions_total{answered="true"}' in response.text


# --- LLM duration histogram ---

def llm_calls() -> float:
    return sample_value(LLM_DURATION, "tax_assistant_llm_duration_seconds_count")


def llm_calls_in_fastest_bucket() -> float:
    return sample_value(LLM_DURATION, "tax_assistant_llm_duration_seconds_bucket", {"le": "0.5"})


async def test_llm_call_is_timed(settings, documents, make_vector_store):
    graph = build_graph(make_vector_store([(documents[0], 0.72)]), FakeListChatModel(responses=["Válasz"]), settings)
    before, fastest_before = llm_calls(), llm_calls_in_fastest_bucket()

    await graph.ainvoke({"question": QUESTION["question"], "top_k": 4})

    assert llm_calls() == before + 1
    assert llm_calls_in_fastest_bucket() == fastest_before + 1


async def test_no_llm_call_is_timed_below_threshold(settings, documents, make_vector_store):
    graph = build_graph(make_vector_store([(documents[0], 0.30)]), FakeListChatModel(responses=[]), settings)
    before = llm_calls()

    result = await graph.ainvoke({"question": QUESTION["question"], "top_k": 4})

    assert result["answered"] is False
    assert llm_calls() == before


async def test_failed_llm_call_is_timed_too(settings, documents, make_vector_store):
    graph = build_graph(make_vector_store([(documents[0], 0.72)]), FakeListChatModel(responses=[]), settings)
    before = llm_calls()

    with pytest.raises(Exception):
        await graph.ainvoke({"question": QUESTION["question"], "top_k": 4})

    assert llm_calls() == before + 1
