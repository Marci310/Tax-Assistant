class AppError(Exception):
    status_code = 500
    error = "internal_error"

    def __init__(self, message: str = "", headers: dict[str, str] | None = None):
        super().__init__(message)
        self.headers = headers or {}

class AuthenticationError(AppError):
    status_code = 401
    error = "unauthorized"

class LLMTimeoutError(AppError):
    status_code = 504
    error = "llm_timeout"

class LLMProviderError(AppError):
    status_code = 502
    error = "llm_unavailable"

class LLMRateLimitError(AppError):
    status_code = 503
    error = "llm_rate_limited"

class VectorStoreUnavailableError(AppError):
    status_code = 503
    error = "vector_store_unavailable"
