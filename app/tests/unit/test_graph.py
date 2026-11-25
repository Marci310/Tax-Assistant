import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.graph import NO_ANSWER, build_graph


# --- Routing: answer vs no_answer ---

async def test_answers_when_a_hit_is_above_threshold(settings, documents, make_vector_store):
    hit = (documents[0], 0.72)
    store = make_vector_store([hit])
    chat_model = FakeListChatModel(responses=["A válasz [1] alapján..."])
    graph = build_graph(store, chat_model, settings)

    result = await graph.ainvoke({"question": "Mi az elismert költség?", "top_k": 4})

    assert result["answered"] is True
    assert result["answer"] == "A válasz [1] alapján..."
    assert result["documents"] == [hit]
    assert result["language"] == "hu"
    assert store.queries == [("Mi az elismert költség?", 4)]

async def test_no_answer_when_all_hits_below_threshold(settings, documents, make_vector_store):
    hits = [(documents[0], 0.4), (documents[1], 0.3)]
    store = make_vector_store(hits)
    chat_model = FakeListChatModel(responses=[])
    graph = build_graph(store, chat_model, settings)

    result = await graph.ainvoke({"question": "Mi az elismert költség?", "top_k": 4})

    assert result["answered"] is False
    assert result["answer"] == NO_ANSWER["hu"]
    assert result["documents"] == []
    assert result["language"] == "hu"
    assert store.queries == [("Mi az elismert költség?", 4)]

async def test_model_refuses_to_answer(settings, documents, make_vector_store):
    hit = (documents[0], 0.72)
    store = make_vector_store([hit])
    chat_model = FakeListChatModel(responses=[NO_ANSWER["hu"]])
    graph = build_graph(store, chat_model, settings)

    result = await graph.ainvoke({"question": "Mi az elismert költség?", "top_k": 4})

    assert result["answered"] is False
    assert result["answer"] == NO_ANSWER["hu"]
    assert result["documents"] == [hit]
    assert result["language"] == "hu"
    assert store.queries == [("Mi az elismert költség?", 4)]

async def test_no_answer_when_no_hits(settings, make_vector_store):
    store = make_vector_store([])
    chat_model = FakeListChatModel(responses=[])
    graph = build_graph(store, chat_model, settings)

    result = await graph.ainvoke({"question": "How to bake a cake?", "top_k": 4})

    assert result["answered"] is False
    assert result["answer"] == NO_ANSWER["en"]
    assert result["documents"] == []
    assert result["language"] == "en"
    assert store.queries == [("How to bake a cake?", 4)]


# --- Retrieval: min_similarity filter and top_k ---

@pytest.mark.parametrize("scores, kept", [
    ([0.45], [0.45]),
    ([0.72, 0.30], [0.72]),
    ([0.72, 0.50], [0.72, 0.50]),
])
async def test_min_similarity_filter(settings, documents, make_vector_store, scores, kept):
    store = make_vector_store([(documents[i], score) for i, score in enumerate(scores)])
    chat_model = FakeListChatModel(responses=["A válasz [1] alapján..."])
    graph = build_graph(store, chat_model, settings)

    result = await graph.ainvoke({"question": "Mi az elismert költség?", "top_k": 4})

    assert [score for _doc, score in result["documents"]] == kept
    assert result["answered"] is True


async def test_top_k_is_passed_to_retrieval(settings, documents, make_vector_store):
    store = make_vector_store([(doc, 0.7) for doc in documents])
    chat_model = FakeListChatModel(responses=["A válasz [1] alapján..."])
    graph = build_graph(store, chat_model, settings)

    result = await graph.ainvoke({"question": "Mi az elismert költség?", "top_k": 2})

    assert store.queries[0][1] == 2
    assert len(result["documents"]) == 2
