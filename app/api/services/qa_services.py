import asyncio
import logging

import httpx
import ollama
import openai
from langgraph.graph.state import CompiledStateGraph
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

from core.config import Settings
from core.exceptions import LLMProviderError, LLMRateLimitError, LLMTimeoutError, VectorStoreUnavailableError
from core.metrics import QUESTIONS
from models import ReturnMessage, Source

logger = logging.getLogger(__name__)

DEFAULT_RETRY_AFTER_SECONDS = 10


def retry_after(exc: openai.RateLimitError) -> str:
    value = exc.response.headers.get("retry-after", "")
    return value if value.isdigit() else str(DEFAULT_RETRY_AFTER_SECONDS)


class QAService:

    def __init__(self, settings: Settings, graph: CompiledStateGraph):
        self.settings = settings
        self.graph = graph

    async def answer(self, question: str, top_k: int | None = None) -> ReturnMessage:
        top_k = top_k if top_k is not None else self.settings.top_k
        logger.info("Question received (top_k=%d, %d chars)", top_k, len(question))

        try:
            async with asyncio.timeout(self.settings.llm_timeout_seconds):
                result = await self.graph.ainvoke({"question": question, "top_k": top_k})
        except (TimeoutError, httpx.TimeoutException, openai.APITimeoutError) as exc:
            raise LLMTimeoutError(
                f"No answer within {self.settings.llm_timeout_seconds} seconds."
            ) from exc
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(
                "The language model provider is rate limited. Try again later.",
                headers={"Retry-After": retry_after(exc)},
            ) from exc
        except (httpx.HTTPError, openai.APIError, ollama.ResponseError, ConnectionError) as exc:
            raise LLMProviderError("The language model provider returned an error.") from exc

        QUESTIONS.labels(answered=str(result["answered"]).lower()).inc()

        sources = [
            Source(
                name=doc.metadata.get("title", doc.metadata.get("source", "unknown")),
                page_number=doc.metadata.get("page"),
                text=doc.page_content,
                similarity=round(score, 3),
            )
            for doc, score in result.get("documents", [])
        ]
        return ReturnMessage(
            answer=result["answer"],
            answered=result["answered"],
            sources=sources,
            model=self.settings.chat_model,
        )
