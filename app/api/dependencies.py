import secrets
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader
from langgraph.graph.state import CompiledStateGraph

from services.qa_service import QAService
from core.config import Settings, get_settings
from core.exceptions import AuthenticationError
from integrations.providers import get_chat_model, get_vector_store
from rag.graph import build_graph

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    key: Annotated[str | None, Security(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value().encode()
    if key is None or not secrets.compare_digest(key.encode(), expected):
        raise AuthenticationError("Missing or invalid API key.")


@lru_cache
def get_graph() -> CompiledStateGraph:
    settings = get_settings()
    return build_graph(get_vector_store(settings), get_chat_model(settings), settings)


def get_qa_service(
    settings: Annotated[Settings, Depends(get_settings)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> QAService:
    return QAService(settings, graph)
