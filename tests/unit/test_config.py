"""Configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.config import Settings
from app.core.enums import ChunkingStrategy, EmbeddingProvider, Environment, LLMProvider


def test_defaults_are_offline_safe_and_typed() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env is Environment.DEVELOPMENT
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.default_chunking_strategy is ChunkingStrategy.RECURSIVE
    assert settings.embedding_provider is EmbeddingProvider.OPENAI
    assert settings.llm_provider is LLMProvider.OPENAI
    assert not settings.is_production


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (".pdf,.txt", [".pdf", ".txt"]),
        (".PDF, txt", [".pdf", ".txt"]),
        ('[".pdf", ".txt"]', [".pdf", ".txt"]),
    ],
)
def test_allowed_extensions_accepts_comma_list_and_json(raw: str, expected: list[str]) -> None:
    settings = Settings(_env_file=None, allowed_extensions=raw)  # type: ignore[arg-type]
    assert settings.allowed_extensions == expected


def test_allowed_extensions_rejects_empty_list() -> None:
    with pytest.raises(PydanticValidationError):
        Settings(_env_file=None, allowed_extensions=[])


def test_api_prefix_is_normalised() -> None:
    assert Settings(_env_file=None, api_v1_prefix="api/v2/").api_v1_prefix == "/api/v2"
    assert Settings(_env_file=None, api_v1_prefix="/v3").api_v1_prefix == "/v3"


def test_blank_optional_settings_become_none() -> None:
    """Blank env values mean "unset" — Compose passes them through as ``""``."""
    settings = Settings(
        _env_file=None,
        openai_api_key="",
        qdrant_api_key="   ",
        openai_base_url="",
        embedding_dimension="   ",
    )
    assert settings.openai_api_key is None
    assert settings.qdrant_api_key is None
    assert settings.openai_api_key_value is None
    assert settings.openai_base_url is None
    assert settings.embedding_dimension is None


def test_max_upload_size_bytes_is_derived() -> None:
    assert Settings(_env_file=None, max_upload_size_mb=3).max_upload_size_bytes == 3 * 1024 * 1024


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(PydanticValidationError, match="DEFAULT_CHUNK_OVERLAP"):
        Settings(_env_file=None, default_chunk_size=100, default_chunk_overlap=100)


def test_offline_providers_are_refused_in_production() -> None:
    with pytest.raises(PydanticValidationError, match="LLM_PROVIDER=mock"):
        Settings(_env_file=None, app_env="production", llm_provider="mock")

    with pytest.raises(PydanticValidationError, match="EMBEDDING_PROVIDER=hash"):
        Settings(_env_file=None, app_env="production", embedding_provider="hash")


def test_production_accepts_real_providers() -> None:
    settings = Settings(
        _env_file=None,
        app_env="production",
        llm_provider="openai",
        embedding_provider="openai",
        openai_api_key="sk-test",
    )
    assert settings.is_production


def test_require_openai_key_explains_the_offline_alternative() -> None:
    settings = Settings(_env_file=None, openai_api_key=None)

    with pytest.raises(ValueError) as excinfo:
        settings.require_openai_key()

    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert "EMBEDDING_PROVIDER=hash" in message
    assert "LLM_PROVIDER=mock" in message


def test_require_openai_key_returns_the_secret() -> None:
    settings = Settings(_env_file=None, openai_api_key="sk-abc123")
    assert settings.require_openai_key() == "sk-abc123"


def test_cors_origins_accepts_csv_and_json() -> None:
    assert Settings(_env_file=None, cors_origins="a.com,b.com").cors_origins == ["a.com", "b.com"]
    assert Settings(_env_file=None, cors_origins='["https://x.io"]').cors_origins == [
        "https://x.io"
    ]
