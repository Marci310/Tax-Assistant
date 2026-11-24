from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings in one typed object.

    Values come from environment variables (in Docker: the compose `env_file`),
    or from a local `.env` file. Variable names are case-insensitive, so the
    field `llm_provider` is filled from `LLM_PROVIDER`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unknown variables in .env are ignored, not an error
    )

    # --- General ---
    app_name: str = "Tax Assistant"
    debug: bool = False
    testing: bool = False
    environment: Literal["development", "production", "testing"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Provider switch: the only setting needed to change run mode ---
    llm_provider: Literal["openai", "ollama"] = "openai"

    # --- OpenAI (used when llm_provider == "openai") ---
    # SecretStr: the key is hidden when settings are printed or logged.
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Ollama (used when llm_provider == "ollama") ---
    ollama_base_url: str = "http://ollama:11434"
    ollama_chat_model: str = "gemma3:12b"
    ollama_embedding_model: str = "bge-m3"

    # --- Qdrant ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "tax_booklets"

    # --- Documents and ingestion ---
    data_dir: Path = Path("/app/data/raw")
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Retrieval and generation ---
    top_k: int = Field(default=4, ge=1, le=20)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def check_consistency(self) -> "Settings":
        # Fail at startup with a clear message, not later on the first request.
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def collection_name(self) -> str:
        # Each provider's embedding model has a different vector size
        # (OpenAI: 1536, bge-m3: 1024), so they cannot share one collection.
        return f"{self.qdrant_collection}_{self.llm_provider}"


@lru_cache
def get_settings() -> Settings:
    """Create the settings once and reuse them.

    A function instead of a module-level `settings = Settings()` object, so that
    importing this module never fails, and tests can override it
    (FastAPI: app.dependency_overrides[get_settings] = ...).
    """
    return Settings()
