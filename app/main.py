import asyncio
import logging
from http import HTTPStatus
from contextlib import asynccontextmanager
from asgi_correlation_id import CorrelationIdFilter, CorrelationIdMiddleware, correlation_id

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.dependencies import get_graph
from api.main import api_router
from core.config import get_settings
from core.db import close_qdrant_client, get_qdrant_client
from core.exceptions import AppError
from integrations.providers import get_embeddings
from models import ErrorMessage
from rag.ingestion import ingest

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    cid_filter = CorrelationIdFilter(uuid_length=32, default_value="-")
    console_handler = logging.StreamHandler()
    console_handler.addFilter(cid_filter)
    logging.basicConfig(
        handlers=[console_handler],
        level=level,
        format="%(asctime)s %(levelname)s %(name)s [%(correlation_id)s]: %(message)s"
    )
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting with provider=%s, collection=%s", settings.llm_provider, settings.collection_name)
    if settings.api_key is None:
        logger.warning("API_KEY is not set: the API is open to anyone. Set it in production.")

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

    await asyncio.to_thread(get_graph)
    yield
    close_qdrant_client()


app = FastAPI(title="Tax Assistant", lifespan=lifespan)

app.add_middleware(CorrelationIdMiddleware)
app.include_router(api_router)


def error_response(status_code: int, error: str, message: str) -> JSONResponse:
    body = ErrorMessage(error=error, message=message, request_id=correlation_id.get())
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("%s on %s: %s", exc.error, request.url.path, exc, exc_info=exc.__cause__)
    return error_response(exc.status_code, exc.error, str(exc))


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    error = HTTPStatus(exc.status_code).phrase.lower().replace(" ", "_")
    logger.info("%s on %s %s", exc.status_code, request.method, request.url.path)
    response = error_response(exc.status_code, error, str(exc.detail))
    response.headers.update(exc.headers or {})
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0]
    if first["type"] == "json_invalid":
        message = "The request body is not valid JSON."
    else:
        field = ".".join(str(part) for part in first["loc"] if part != "body")
        message = f"{field}: {first['msg']}" if field else first["msg"]
    logger.info("Invalid request on %s: %s", request.url.path, message)
    return error_response(422, "invalid_request", message)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    response = error_response(500, "internal_error", "An unexpected error occurred.")
    response.headers["X-Request-ID"] = correlation_id.get() or ""
    return response
