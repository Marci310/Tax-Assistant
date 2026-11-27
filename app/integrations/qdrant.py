import logging
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from core.config import get_settings

logger = logging.getLogger(__name__)


class VectorStoreConfigError(RuntimeError):
    """The existing collection does not match the current configuration."""


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
    return QdrantClient(url=settings.qdrant_url, timeout=10)


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> bool:
    if client.collection_exists(name):
        vectors = client.get_collection(name).config.params.vectors
        existing_size = vectors.size if isinstance(vectors, models.VectorParams) else None
        if existing_size != vector_size:
            raise VectorStoreConfigError(
                f"Collection '{name}' has vector size {existing_size}, but the current "
                f"embedding model produces {vector_size}. Delete the collection or "
                f"change QDRANT_COLLECTION, then ingest again."
            )
        return False

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info("Created collection '%s' (vector size %d)", name, vector_size)
    return True


def store_documents(
    client: QdrantClient,
    name: str,
    embeddings: Embeddings,
    documents: list[Document],
    ids: list[str],
) -> None:
    store = QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)
    store.add_documents(documents, ids=ids, batch_size=64)


def count_points(client: QdrantClient, name: str) -> int:
    if not client.collection_exists(name):
        return 0
    return client.count(collection_name=name, exact=True).count


def is_qdrant_healthy(client: QdrantClient) -> bool:
    try:
        client.get_collections()
        return True
    except Exception:
        logger.warning("Qdrant health check failed", exc_info=True)
        return False


def close_qdrant_client() -> None:
    if get_qdrant_client.cache_info().currsize:
        get_qdrant_client().close()
        get_qdrant_client.cache_clear()
