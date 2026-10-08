"""Text-extraction abstractions.

Extractors are synchronous (they are CPU/IO bound) and the ingestion service
runs them in a worker thread so the event loop is never blocked.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

#: Collapse runs of 3+ blank lines into a single blank line and normalise EOLs.
_WHITESPACE_RE = re.compile(r"[ \t]+\n")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalise_text(text: str) -> str:
    """Normalise line endings and whitespace without destroying structure.

    Paragraph boundaries (``\\n\\n``) are preserved because the recursive
    chunker relies on them.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub("\n", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """The result of extracting text from an uploaded file."""

    text: str
    page_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class TextExtractor(ABC):
    """Extracts plain text from a binary upload."""

    #: Lower-cased file extensions this extractor handles, e.g. ``{".pdf"}``.
    extensions: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def extract(self, *, filename: str, data: bytes) -> ExtractedDocument:
        """Return the extracted text for ``data``.

        Raises:
            ExtractionError: if the payload cannot be parsed or yields no text.
        """

    def supports(self, extension: str) -> bool:
        return extension.lower() in self.extensions
