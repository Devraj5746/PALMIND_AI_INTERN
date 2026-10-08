"""Document text extraction (PDF, plain text)."""

from __future__ import annotations

from app.services.extraction.base import (
    ExtractedDocument,
    TextExtractor,
    normalise_text,
)
from app.services.extraction.pdf_extractor import PdfTextExtractor
from app.services.extraction.registry import ExtractorRegistry
from app.services.extraction.txt_extractor import PlainTextExtractor

__all__ = [
    "ExtractedDocument",
    "ExtractorRegistry",
    "PdfTextExtractor",
    "PlainTextExtractor",
    "TextExtractor",
    "normalise_text",
]
