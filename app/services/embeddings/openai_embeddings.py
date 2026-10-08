"""OpenAI embedding provider (thin wrapper over ``langchain_openai``).

``langchain_openai.OpenAIEmbeddings`` already handles batching, retries and true
async I/O, so this provider only adds the metadata the rest of the application
needs (a stable name and the vector dimension) plus consistent error mapping.
"""

from __future__ import annotations

from typing import ClassVar

import openai
from langchain_openai import OpenAIEmbeddings

from app.core.errors import ConfigurationError, UpstreamProviderError
from app.core.logging import get_logger
from app.services.embeddings.base import BaseEmbeddingProvider

logger = get_logger(__name__)

#: Known output dimensions per OpenAI embedding model.
MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingsProvider(BaseEmbeddingProvider):
    """Embeddings served by the OpenAI API."""

    name: ClassVar[str] = "openai"

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str,
        base_url: str | None = None,
        batch_size: int = 64,
        dimension: int | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        resolved_dimension = dimension or MODEL_DIMENSIONS.get(model)
        if resolved_dimension is None:
            raise ConfigurationError(
                f"Unknown embedding model '{model}'.",
                details={
                    "reason": "Cannot infer the vector dimension.",
                    "known_models": sorted(MODEL_DIMENSIONS),
                    "hint": "Set EMBEDDING_DIMENSION to the model's output size.",
                },
            )

        super().__init__(model=model, dimension=resolved_dimension)

        # ``langchain-openai`` declares the legacy ``openai_*`` field names; the
        # short ``api_key``/``base_url`` spellings are aliases that mypy does not
        # see, so the declared names are used here.
        self._client = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            chunk_size=batch_size,
            max_retries=max_retries,
            request_timeout=timeout,
            # Only the v3 models accept a reduced output size.
            dimensions=resolved_dimension if model.startswith("text-embedding-3") else None,
        )

    # -- Embeddings interface ---------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._client.embed_documents(texts)
        except openai.OpenAIError as exc:
            raise self._map_error(exc, "embed_documents") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._client.embed_query(text)
        except openai.OpenAIError as exc:
            raise self._map_error(exc, "embed_query") from exc

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return await self._client.aembed_documents(texts)
        except openai.OpenAIError as exc:
            raise self._map_error(exc, "aembed_documents") from exc

    async def aembed_query(self, text: str) -> list[float]:
        try:
            return await self._client.aembed_query(text)
        except openai.OpenAIError as exc:
            raise self._map_error(exc, "aembed_query") from exc

    async def health(self) -> None:
        try:
            await self._client.aembed_query("health check")
        except openai.OpenAIError as exc:
            raise self._map_error(exc, "health") from exc

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _map_error(exc: Exception, operation: str) -> UpstreamProviderError:
        logger.warning("OpenAI embedding call '%s' failed: %s", operation, exc)
        return UpstreamProviderError(
            "The embedding provider could not process the request.",
            details={"operation": operation, "reason": str(exc)},
        )
