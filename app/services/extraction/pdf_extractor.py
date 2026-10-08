"""PDF text extraction backed by ``pypdf``."""

from __future__ import annotations

from io import BytesIO
from typing import ClassVar

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.errors import ExtractionError
from app.core.logging import get_logger
from app.services.extraction.base import ExtractedDocument, TextExtractor, normalise_text

logger = get_logger(__name__)


class PdfTextExtractor(TextExtractor):
    """Extract text from every page of a PDF document."""

    extensions: ClassVar[frozenset[str]] = frozenset({".pdf"})

    def extract(self, *, filename: str, data: bytes) -> ExtractedDocument:
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                # Attempt the common "empty password" case before giving up.
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise ExtractionError(
                        "The PDF is password-protected and cannot be read.",
                        details={"filename": filename},
                    ) from exc

            pages: list[str] = []
            per_page_chars: list[int] = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                per_page_chars.append(len(page_text))
                pages.append(page_text)

            raw_text = "\n\n".join(pages)
            text = normalise_text(raw_text)
            page_count = len(reader.pages)

        except ExtractionError:
            raise
        except PdfReadError as exc:
            raise ExtractionError(
                "The uploaded file is not a valid PDF.",
                details={"filename": filename, "reason": str(exc)},
            ) from exc
        except Exception as exc:
            logger.warning("Unexpected PDF parse failure for %s: %s", filename, exc)
            raise ExtractionError(
                "The PDF could not be parsed.",
                details={"filename": filename, "reason": str(exc)},
            ) from exc

        if not text:
            raise ExtractionError(
                "No extractable text found in the PDF. It may be a scanned "
                "image without an OCR layer.",
                details={"filename": filename, "page_count": page_count},
            )

        return ExtractedDocument(
            text=text,
            page_count=page_count,
            metadata={
                "extractor": "pypdf",
                "page_count": page_count,
                "per_page_chars": per_page_chars,
            },
        )
