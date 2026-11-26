from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from api.routes.ask import router as ask_router
from api.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(ask_router, prefix="/api/v1", tags=["ask"], dependencies=[Depends(verify_api_key)])
api_router.include_router(health_router, tags=["health"])
