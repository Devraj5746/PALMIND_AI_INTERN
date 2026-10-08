"""Offline, deterministic embedding provider.

This is the "hashing trick" vectoriser: every token (and token bigram) is hashed
to a bucket and accumulated with a sign derived from the hash.  Vectors are
L2-normalised, so cosine similarity behaves sensibly.

It captures **lexical overlap**, not semantics — a query only matches a chunk
that shares vocabulary.  That is deliberate: it makes the whole stack runnable
with zero API keys and makes retrieval assertions in the test-suite completely
deterministic (the hash is stable across processes because it uses blake2b, not
Python's randomised ``hash()``).

For production, use ``EMBEDDING_PROVIDER=openai``.
"""

from __future__ import annotations

import hashlib
import math
import re
from itertools import pairwise
from typing import ClassVar

from app.services.embeddings.base import BaseEmbeddingProvider

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stable_hash(token: str) -> int:
    """Return a process-stable integer hash for ``token``."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


class HashEmbeddings(BaseEmbeddingProvider):
    """Deterministic local embeddings for offline use and CI."""

    name: ClassVar[str] = "hash"

    def __init__(self, *, dimension: int = 384, model: str = "hash-blake2b-v1") -> None:
        super().__init__(model=model, dimension=dimension)

    # -- Embeddings interface ---------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorise(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorise(text)

    # -- internals ---------------------------------------------------------
    def _tokenise(self, text: str) -> list[str]:
        tokens = TOKEN_RE.findall(text.lower())
        # Add bigrams so that short phrases carry slightly more signal.
        bigrams = [f"{first}_{second}" for first, second in pairwise(tokens)]
        return tokens + bigrams

    def _vectorise(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokenise(text):
            digest = _stable_hash(token)
            index = digest % self.dimension
            sign = 1.0 if (digest >> 63) & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
