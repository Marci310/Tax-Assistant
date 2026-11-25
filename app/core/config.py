from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- General ---
    app_name: str = "Tax Assistant"
    debug: bool = False
    testing: bool = False
    environment: Literal["development", "production", "testing"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Provider settings ---
    llm_provider: Literal["openai", "ollama"] = "openai"

    # --- OpenAI ---
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Ollama ---
    ollama_base_url: str = "http://ollama:11434"
    ollama_chat_model: str = "gemma3:12b"
    ollama_embedding_model: str = "bge-m3"

    # --- Qdrant ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "tax_booklets"

    # --- Documents and ingestion ---
    data_dir: Path = Path("/app/data/raw")
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    ingest_on_startup: bool = True

    # --- RAG ---
    top_k: int = Field(default=4, ge=1, le=20)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    min_similarity: float = Field(default=0.45, ge=0, le=1)
    max_concurrent_llm_calls: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def check_consistency(self) -> "Settings":
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def collection_name(self) -> str:
        return f"{self.qdrant_collection}_{self.llm_provider}"

    @property
    def chat_model(self) -> str:
        return self.openai_chat_model if self.llm_provider == "openai" else self.ollama_chat_model


@lru_cache
def get_settings() -> Settings:

    return Settings()
