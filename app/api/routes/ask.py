from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_qa_service
from api.services.qa_services import QAService
from models import ErrorMessage, MessageIn, ReturnMessage

router = APIRouter()

@router.post(
    "/ask",
    response_model=ReturnMessage,
    responses={
        401: {"model": ErrorMessage, "description": "Missing or invalid API key."},
        422: {"model": ErrorMessage, "description": "Invalid input, e.g. an empty question."},
        502: {"model": ErrorMessage, "description": "The LLM provider returned an error."},
        503: {"model": ErrorMessage, "description": "The vector store or LLM is unavailable."},
        504: {"model": ErrorMessage, "description": "The LLM did not answer in time."},
    },
)
async def ask(
    message: MessageIn,
    service: Annotated[QAService, Depends(get_qa_service)],
) -> ReturnMessage:
    return await service.answer(question=message.question, top_k=message.top_k)
