from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph

from api.services.qa_services import QAService
from core.config import Settings, get_settings
from integrations.providers import get_chat_model, get_vector_store
from rag.graph import build_graph


@lru_cache
def get_graph() -> CompiledStateGraph:
    settings = get_settings()
    return build_graph(get_vector_store(settings), get_chat_model(settings), settings)


def get_qa_service(
    settings: Annotated[Settings, Depends(get_settings)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> QAService:
    return QAService(settings, graph)


QAServiceDep = Annotated[QAService, Depends(get_qa_service)]
