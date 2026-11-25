import logging

from langgraph.graph.state import CompiledStateGraph

from core.config import Settings
from models import ReturnMessage, Source

logger = logging.getLogger(__name__)


class QAService:

    def __init__(self, settings: Settings, graph: CompiledStateGraph):
        self.settings = settings
        self.graph = graph

    async def answer(self, question: str, top_k: int | None = None) -> ReturnMessage:
        top_k = top_k if top_k is not None else self.settings.top_k
        logger.info("Question received (top_k=%d, %d chars)", top_k, len(question))

        result = await self.graph.ainvoke({"question": question, "top_k": top_k})

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
