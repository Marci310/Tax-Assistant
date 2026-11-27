from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


def get_datetime_utc() -> datetime:
    return datetime.now(UTC)

QuestionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=1000),
]


class MessageIn(BaseModel):
    question: QuestionText = Field(description="The user's question, in Hungarian or English.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="How many passages to retrieve. Uses the server default if omitted.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "Ki a társasági adó alanya?"},
                {"question": "Mikor kell benyújtani a társasági adó bevallását?", "top_k": 5},
            ]
        }
    )


class Source(BaseModel):
    name: str = Field(description="Title of the booklet.")
    page_number: int | None = Field(default=None, ge=1)
    text: str = Field(description="Excerpt of the passage the answer used.")
    similarity: float | None = Field(default=None, description="Cosine similarity, higher is closer.")


class ReturnMessage(BaseModel):
    answer: str
    answered: bool = Field(description="False if the booklets do not cover the question.")
    sources: list[Source] = Field(default_factory=list)
    model: str = Field(description="The chat model that wrote the answer.")
    timestamp: datetime = Field(default_factory=get_datetime_utc)


class ErrorMessage(BaseModel):
    error: str = Field(description="Machine-readable code, e.g. invalid_request, llm_timeout.")
    message: str = Field(description="Human-readable explanation.")
    request_id: str | None = Field(default=None, description="Matches the server log entry.")
    timestamp: datetime = Field(default_factory=get_datetime_utc)


class HealthCheck(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant_healthy: bool
    indexed_chunks: int = Field(ge=0)
    provider: Literal["openai", "ollama"]
    timestamp: datetime = Field(default_factory=get_datetime_utc)
