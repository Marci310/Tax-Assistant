from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from core.config import Settings


@lru_cache
def get_embeddings(settings: Settings) -> Embeddings:
    if settings.llm_provider == "openai":
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=3,
        )
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.llm_timeout_seconds},
    )


@lru_cache
def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=3,
            temperature=0,
        )
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.llm_timeout_seconds},
        temperature=0,
    )
