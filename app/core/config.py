"""Typed application settings loaded from environment variables / ``.env``.

A single :class:`Settings` instance is the source of truth for every tunable in
the application.  Nothing else in the codebase should read ``os.environ``
directly, which keeps configuration testable and explicit.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.enums import ChunkingStrategy, EmbeddingProvider, Environment, LLMProvider

LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Runtime configuration for the backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Application --------------------------------------------------------
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: LogLevelName = "INFO"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # -- LLM ----------------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=1024, gt=0)
    llm_request_timeout_seconds: float = Field(default=60.0, gt=0)

    # -- Embeddings ---------------------------------------------------------
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = Field(default=64, gt=0)
    hash_embedding_dimension: int = Field(
        default=384,
        gt=0,
        description="Vector size used by the offline hash embedding provider.",
    )
    embedding_dimension: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Explicit vector size. Only needed for embedding models that are not in "
            "the built-in model→dimension table; otherwise it is inferred."
        ),
    )

    # -- Qdrant -------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"
    qdrant_prefer_grpc: bool = False

    # -- Redis --------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    chat_memory_ttl_seconds: int = Field(default=86_400, gt=0)
    chat_memory_max_messages: int = Field(default=20, gt=0)

    # -- Database -----------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./rag.db"
    db_echo: bool = False

    # -- Ingestion ----------------------------------------------------------
    max_upload_size_mb: int = Field(default=25, gt=0)
    allowed_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".txt"])
    default_chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    default_chunk_size: int = Field(default=1000, gt=0)
    default_chunk_overlap: int = Field(default=150, ge=0)

    # -- Retrieval ----------------------------------------------------------
    rag_top_k: int = Field(default=4, gt=0, le=50)
    rag_score_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    rag_max_context_chars: int = Field(default=8000, gt=0)
    rag_condense_followups: bool = True

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _parse_extensions(cls, value: Any) -> Any:
        """Accept ``[".pdf", ".txt"]``, ``.pdf,.txt`` or a JSON string."""
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):
                parsed = json.loads(raw)
                return [str(item) for item in parsed]
            return [part.strip() for part in raw.split(",") if part.strip()]
        return value

    @field_validator("allowed_extensions")
    @classmethod
    def _normalise_extensions(cls, value: list[str]) -> list[str]:
        normalised = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in value}
        if not normalised:
            raise ValueError("allowed_extensions must contain at least one extension")
        return sorted(normalised)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, value: Any) -> Any:
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):
                return json.loads(raw)
            return [part.strip() for part in raw.split(",") if part.strip()]
        return value

    @field_validator("api_v1_prefix")
    @classmethod
    def _normalise_prefix(cls, value: str) -> str:
        prefix = value.strip()
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        return prefix.rstrip("/")

    @field_validator(
        "openai_api_key",
        "qdrant_api_key",
        "openai_base_url",
        "embedding_dimension",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        """Treat a blank optional setting as unset instead of as ``""``.

        Compose-driven deployments hit this: an env-file entry with an empty
        value is passed through as an empty string (only a *non-empty* value
        gets its trailing comment stripped), so a blank ``EMBEDDING_DIMENSION``
        or ``OPENAI_BASE_URL`` must fall back to its default rather than fail
        validation or be used verbatim.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _validate_combinations(self) -> Settings:
        if self.default_chunk_overlap >= self.default_chunk_size:
            raise ValueError(
                "DEFAULT_CHUNK_OVERLAP must be smaller than DEFAULT_CHUNK_SIZE "
                f"(got overlap={self.default_chunk_overlap}, "
                f"size={self.default_chunk_size})",
            )
        if self.app_env == Environment.PRODUCTION:
            if self.llm_provider == LLMProvider.MOCK:
                raise ValueError("LLM_PROVIDER=mock is not allowed when APP_ENV=production")
            if self.embedding_provider == EmbeddingProvider.HASH:
                raise ValueError("EMBEDDING_PROVIDER=hash is not allowed when APP_ENV=production")
        return self

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def openai_api_key_value(self) -> str | None:
        """Plain-text OpenAI key, or ``None`` when unset."""
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    @property
    def qdrant_api_key_value(self) -> str | None:
        return self.qdrant_api_key.get_secret_value() if self.qdrant_api_key else None

    def require_openai_key(self) -> str:
        """Return the OpenAI key or raise a helpful error explaining how to set it."""
        key = self.openai_api_key_value
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is required for the openai providers. "
                "Set it in your .env file, or use EMBEDDING_PROVIDER=hash and "
                "LLM_PROVIDER=mock for a fully offline setup.",
            )
        return key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
