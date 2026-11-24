from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.main import api_router
from core.config import get_settings
from core.db import close_qdrant_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()
    yield
    close_qdrant_client()


app = FastAPI(title="Tax Assistant", lifespan=lifespan)
app.include_router(api_router)
