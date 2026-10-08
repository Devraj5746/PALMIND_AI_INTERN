"""Extension-based dispatch to the right :class:`TextExtractor`."""

from __future__ import annotations

from pathlib import PurePosixPath

from app.core.errors import UnsupportedMediaTypeError
from app.services.extraction.base import TextExtractor
from app.services.extraction.pdf_extractor import PdfTextExtractor
from app.services.extraction.txt_extractor import PlainTextExtractor


class ExtractorRegistry:
    """Maps a filename to the extractor capable of handling it."""

    def __init__(self, extractors: list[TextExtractor] | None = None) -> None:
        self._extractors: list[TextExtractor] = extractors or [
            PdfTextExtractor(),
            PlainTextExtractor(),
        ]

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset().union(*(ex.extensions for ex in self._extractors))

    @staticmethod
    def extension_of(filename: str) -> str:
        """Return the lower-cased extension of ``filename`` (``.pdf``, ``.txt``)."""
        return PurePosixPath(filename.replace("\\", "/")).suffix.lower()

    def get(self, filename: str) -> TextExtractor:
        """Return the extractor for ``filename``.

        Raises:
            UnsupportedMediaTypeError: when no extractor matches the extension.
        """
        extension = self.extension_of(filename)
        if not extension:
            raise UnsupportedMediaTypeError(
                "The uploaded file has no extension and its type cannot be inferred.",
                details={
                    "filename": filename,
                    "supported_extensions": sorted(self.supported_extensions),
                },
            )

        for extractor in self._extractors:
            if extractor.supports(extension):
                return extractor

        raise UnsupportedMediaTypeError(
            f"Files of type '{extension}' are not supported.",
            details={
                "filename": filename,
                "supported_extensions": sorted(self.supported_extensions),
            },
        )
