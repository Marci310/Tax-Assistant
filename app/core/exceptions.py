class AppError(Exception):
    status_code = 500
    error = "internal_error"

class LLMTimeoutError(AppError):
    status_code = 504
    error = "llm_timeout"

class LLMProviderError(AppError):
    status_code = 502
    error = "llm_unavailable"

class VectorStoreUnavailableError(AppError):
    status_code = 503
    error = "vector_store_unavailable"