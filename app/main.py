import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.main import api_router
from core.config import get_settings
from core.db import close_qdrant_client, get_qdrant_client
from integrations.providers import get_embeddings
from rag.ingestion import ingest

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting with provider=%s, collection=%s", settings.llm_provider, settings.collection_name)

    if settings.ingest_on_startup:
        try:
            chunks = await asyncio.to_thread(
                ingest, settings, get_qdrant_client(), get_embeddings(settings)
            )
        except Exception:
            logger.exception("Ingestion failed, stopping the app")
            raise
        logger.info("Ready: %d chunks indexed", chunks)
    else:
        logger.info("INGEST_ON_STARTUP=false, skipping ingestion")

    yield
    close_qdrant_client()


app = FastAPI(title="Tax Assistant", lifespan=lifespan)
app.include_router(api_router)
