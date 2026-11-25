from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from qdrant_client import QdrantClient

from core.config import Settings, get_settings
from core.db import count_points, get_qdrant_client, is_qdrant_healthy
from models import HealthCheck

router = APIRouter()
@router.get(
    "/health",
    response_model=HealthCheck,
    responses={503: {"model": HealthCheck, "description": "A dependency is unavailable."}},
)
def health_check(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    client: Annotated[QdrantClient, Depends(get_qdrant_client)],
) -> HealthCheck:
    qdrant_healthy = is_qdrant_healthy(client)
    indexed_chunks = count_points(client, settings.collection_name) if qdrant_healthy else 0

    if not qdrant_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthCheck(
        status="ok" if qdrant_healthy else "degraded",
        qdrant_healthy=qdrant_healthy,
        indexed_chunks=indexed_chunks,
        provider=settings.llm_provider,
    )
