import pytest
from pydantic import SecretStr

from core.config import Settings


@pytest.mark.parametrize("provider, expected", [("openai", 0.32), ("ollama", 0.45)])
def test_min_similarity_follows_provider(provider, expected):
    settings = Settings(
        _env_file=None,
        llm_provider=provider,
        openai_api_key=SecretStr("test-key"),
        openai_min_similarity=0.32,
        ollama_min_similarity=0.45,
    )

    assert settings.min_similarity == expected
