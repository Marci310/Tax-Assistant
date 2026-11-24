from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_qa_service
from api.services.qa_services import QAService
from models import ErrorMessage, MessageIn, ReturnMessage

router = APIRouter()

@router.post(
    "/ask",
    response_model=ReturnMessage
)
async def ask(
    message: MessageIn,
    service: Annotated[QAService, Depends(get_qa_service)],
) -> ReturnMessage:
    return await service.answer(question=message.question, top_k=message.top_k)
