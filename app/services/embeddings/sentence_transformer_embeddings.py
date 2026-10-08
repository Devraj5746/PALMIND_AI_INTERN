"""Optional local embedding provider backed by ``sentence-transformers``.

Selected with ``EMBEDDING_PROVIDER=sentence_transformers`` and installed via the
``local-embeddings`` extra::

    uv pip install -e ".[local-embeddings]"

Inference is synchronous and CPU-bound, so LangChain's default async bridge runs
it in a worker thread — the event loop is never blocked.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.core.errors import ConfigurationError
from app.core.logging import get_logger
from app.services.embeddings.base import BaseEmbeddingProvider

logger = get_logger(__name__)

#: Common model -> output dimension map (avoids loading the model to find out).
MODEL_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
}


class SentenceTransformerEmbeddings(BaseEmbeddingProvider):
    """Local, self-hosted embeddings with no network calls at inference time."""

    name: ClassVar[str] = "sentence_transformers"

    def __init__(
        self,
        *,
        model: str = "all-MiniLM-L6-v2",
        dimension: int | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ConfigurationError(
                "The 'sentence_transformers' embedding provider requires the "
                "'sentence-transformers' package.",
                details={"hint": 'uv pip install -e ".[local-embeddings]"'},
            ) from exc

        self._model = SentenceTransformer(model, device=device)
        resolved_dimension = (
            dimension
            or MODEL_DIMENSIONS.get(model)
            or int(self._model.get_sentence_embedding_dimension())
        )
        super().__init__(model=model, dimension=resolved_dimension)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._to_float_lists(vectors)

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._to_float_lists(vector)[0]

    @staticmethod
    def _to_float_lists(vectors: Any) -> list[list[float]]:
        return [[float(value) for value in row] for row in vectors]
