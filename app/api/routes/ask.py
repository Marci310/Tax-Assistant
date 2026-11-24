from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_qa_service
from api.services.qa_services import QAService
from models import ErrorMessage, MessageIn, ReturnMessage

router = APIRouter()


# `async def`: the service will await the LLM / LangGraph calls, so nothing here blocks.
# The route only does HTTP work: validated input in, service call, response out.
# The `responses` entries only document the error format in /docs for now;
# the exception handlers that actually produce them come later.
@router.post(
    "/ask",
    response_model=ReturnMessage,
)
async def ask(
    message: MessageIn,
    service: Annotated[QAService, Depends(get_qa_service)],
) -> ReturnMessage:
    return await service.answer(question=message.question, top_k=message.top_k)
