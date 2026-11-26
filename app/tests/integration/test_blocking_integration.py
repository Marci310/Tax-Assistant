import pytest

from api.dependencies import get_graph

pytestmark = pytest.mark.integration

@pytest.fixture
def graph():
    return get_graph()

async def test_graph(graph):
    response = await graph.ainvoke({"question": "Mi a növekedési adóhitel?", "top_k": 4})
    assert response["answered"] is True