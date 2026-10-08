"""Plain-text extraction with pragmatic encoding fallbacks."""

from __future__ import annotations

from typing import ClassVar

from app.core.errors import ExtractionError
from app.core.logging import get_logger
from app.services.extraction.base import ExtractedDocument, TextExtractor, normalise_text

logger = get_logger(__name__)

#: Tried in order; the first codec that decodes without replacement wins.
ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1")


class PlainTextExtractor(TextExtractor):
    """Decode a ``.txt`` upload into normalised Unicode text."""

    extensions: ClassVar[frozenset[str]] = frozenset({".txt"})

    def extract(self, *, filename: str, data: bytes) -> ExtractedDocument:
        if not data:
            raise ExtractionError("The uploaded file is empty.", details={"filename": filename})

        text, encoding = self._decode(data)
        text = normalise_text(text)

        if not text:
            raise ExtractionError(
                "The uploaded file contains no text.",
                details={"filename": filename},
            )

        return ExtractedDocument(
            text=text,
            page_count=None,
            metadata={"extractor": "plaintext", "encoding": encoding},
        )

    @staticmethod
    def _decode(data: bytes) -> tuple[str, str]:
        """Return ``(text, encoding)`` using the first lossless codec."""
        for encoding in ENCODING_CANDIDATES:
            try:
                return data.decode(encoding), encoding
            except (UnicodeDecodeError, LookupError):
                continue

        logger.warning("Falling back to lossy latin-1 decoding for text upload")
        return data.decode("latin-1", errors="replace"), "latin-1(replace)"
