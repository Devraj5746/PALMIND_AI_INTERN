"""Factory that builds the configured embedding provider."""

from __future__ import annotations

from app.core.config import Settings
from app.core.enums import EmbeddingProvider
from app.core.errors import ConfigurationError
from app.core.logging import get_logger
from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.embeddings.hash_embeddings import HashEmbeddings

logger = get_logger(__name__)


def build_embedding_provider(settings: Settings) -> BaseEmbeddingProvider:
    """Instantiate the provider selected by ``EMBEDDING_PROVIDER``.

    Raises:
        ConfigurationError: when the selection is unsupported or the required
            credentials / optional dependency are missing.
    """
    provider = settings.embedding_provider

    if provider == EmbeddingProvider.OPENAI:
        # Imported lazily so the offline providers never require the openai SDK.
        from app.services.embeddings.openai_embeddings import OpenAIEmbeddingsProvider

        return OpenAIEmbeddingsProvider(
            model=settings.embedding_model,
            api_key=settings.require_openai_key(),
            base_url=settings.openai_base_url,
            batch_size=settings.embedding_batch_size,
            dimension=settings.embedding_dimension,
            timeout=settings.llm_request_timeout_seconds,
        )

    if provider == EmbeddingProvider.HASH:
        logger.warning(
            "Using the offline 'hash' embedding provider: retrieval relies on "
            "lexical overlap, not semantics. Set EMBEDDING_PROVIDER=openai for production.",
        )
        return HashEmbeddings(dimension=settings.hash_embedding_dimension)

    if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        from app.services.embeddings.sentence_transformer_embeddings import (
            SentenceTransformerEmbeddings,
        )

        return SentenceTransformerEmbeddings(
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
        )

    raise ConfigurationError(  # pragma: no cover - guarded by the enum
        f"Unsupported embedding provider '{provider}'.",
        details={"supported_providers": [p.value for p in EmbeddingProvider]},
    )


__all__ = ["build_embedding_provider"]
