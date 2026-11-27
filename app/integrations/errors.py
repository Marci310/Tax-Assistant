import httpx
import ollama
import openai
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    httpx.TimeoutException,
    openai.APITimeoutError,
)

VECTOR_STORE_ERRORS: tuple[type[BaseException], ...] = (
    ResponseHandlingException,
    UnexpectedResponse,
)

RATE_LIMIT_ERRORS: tuple[type[BaseException], ...] = (openai.RateLimitError,)

PROVIDER_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    openai.APIError,
    ollama.ResponseError,
    ConnectionError,
)

DEFAULT_RETRY_AFTER_SECONDS = 10


def retry_after_seconds(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after", "") if response is not None else ""
    return value if value.isdigit() else str(DEFAULT_RETRY_AFTER_SECONDS)
