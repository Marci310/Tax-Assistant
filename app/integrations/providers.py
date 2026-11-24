from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from functools import lru_cache

from core.config import Settings

@lru_cache
def get_embeddings(settings: Settings) -> Embeddings:
    if settings.llm_provider == "openai":
        return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds, max_retries=3)
    return OllamaEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url)

@lru_cache
def get_chat_model(settings: Settings):
    if settings.chat_model == "openai":
        return ChatOpenAI(model=settings.openai_chat_model, api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds, max_retries=3)
    return ChatOllama(model=settings.ollama_chat_model, base_url=settings.ollama_base_url)