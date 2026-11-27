import asyncio
import logging

from langgraph.graph.state import CompiledStateGraph

from core.config import Settings
from core.exceptions import LLMProviderError, LLMRateLimitError, LLMTimeoutError, VectorStoreUnavailableError
from core.metrics import QUESTIONS
from api.schemas import ReturnMessage, Source
from integrations.errors import (
    PROVIDER_ERRORS,
    RATE_LIMIT_ERRORS,
    TIMEOUT_ERRORS,
    VECTOR_STORE_ERRORS,
    retry_after_seconds,
)

logger = logging.getLogger(__name__)


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
        except TIMEOUT_ERRORS as exc:
            raise LLMTimeoutError(
                f"No answer within {self.settings.llm_timeout_seconds} seconds."
            ) from exc
        except VECTOR_STORE_ERRORS as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable.") from exc
        except RATE_LIMIT_ERRORS as exc:
            raise LLMRateLimitError(
                "The language model provider is rate limited. Try again later.",
                headers={"Retry-After": retry_after_seconds(exc)},
            ) from exc
        except PROVIDER_ERRORS as exc:
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
