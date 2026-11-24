import logging

from core.config import Settings
from models import ReturnMessage

logger = logging.getLogger(__name__)


class QAService:

    def __init__(self, settings: Settings):
        self.settings = settings

    async def answer(self, question: str, top_k: int | None = None) -> ReturnMessage:
        top_k = top_k if top_k is not None else self.settings.top_k
        logger.info("Question received (top_k=%d, %d chars)", top_k, len(question))

        return ReturnMessage(
            answer="Not implemented yet.",
            answered=False,
            sources=[],
            model=self.settings.chat_model,
        )
