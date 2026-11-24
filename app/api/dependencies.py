from typing import Annotated

from fastapi import Depends

from api.services.qa_services import QAService
from core.config import Settings, get_settings


def get_qa_service(settings: Annotated[Settings, Depends(get_settings)]) -> QAService:
    return QAService(settings)


QAServiceDep = Annotated[QAService, Depends(get_qa_service)]
