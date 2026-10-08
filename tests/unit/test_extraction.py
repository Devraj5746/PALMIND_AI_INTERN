"""Text extraction for ``.txt`` and ``.pdf`` uploads."""

from __future__ import annotations

import pytest

from app.core.errors import ExtractionError, UnsupportedMediaTypeError
from app.services.extraction.base import normalise_text
from app.services.extraction.pdf_extractor import PdfTextExtractor
from app.services.extraction.registry import ExtractorRegistry
from app.services.extraction.txt_extractor import PlainTextExtractor
from tests.helpers import build_pdf

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalise_text_unifies_line_endings() -> None:
    assert normalise_text("a\r\nb\rc") == "a\nb\nc"


def test_normalise_text_collapses_blank_line_runs_but_keeps_paragraphs() -> None:
    assert normalise_text("a\n\n\n\n\nb") == "a\n\nb"


def test_normalise_text_strips_trailing_spaces_and_outer_whitespace() -> None:
    assert normalise_text("  hello   \n\n  world  ") == "hello\n\n  world"


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def test_text_extractor_decodes_utf8() -> None:
    result = PlainTextExtractor().extract(filename="a.txt", data="caf\u00e9 n\u00e4ive".encode())

    assert result.text == "caf\u00e9 n\u00e4ive"
    assert result.page_count is None
    # ``utf-8-sig`` is tried first so a byte-order mark never leaks into the text.
    assert result.metadata["encoding"] == "utf-8-sig"


def test_text_extractor_strips_a_utf8_bom() -> None:
    result = PlainTextExtractor().extract(filename="a.txt", data=b"\xef\xbb\xbfhello")
    assert result.text == "hello"


def test_text_extractor_falls_back_to_cp1252() -> None:
    # 0x92 is a curly apostrophe in cp1252 and invalid utf-8.
    result = PlainTextExtractor().extract(filename="a.txt", data=b"it\x92s fine")
    assert "\u2019" in result.text


def test_text_extractor_rejects_empty_payload() -> None:
    with pytest.raises(ExtractionError, match="empty"):
        PlainTextExtractor().extract(filename="a.txt", data=b"")


def test_text_extractor_rejects_whitespace_only_payload() -> None:
    with pytest.raises(ExtractionError, match="no text"):
        PlainTextExtractor().extract(filename="a.txt", data=b"   \n\t  ")


def test_char_count_and_is_empty_helpers() -> None:
    result = PlainTextExtractor().extract(filename="a.txt", data=b"abcd")
    assert result.char_count == 4
    assert result.is_empty is False


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def test_pdf_extractor_reads_every_page() -> None:
    data = build_pdf(["First page about leave.", "Second page about equipment."])
    result = PdfTextExtractor().extract(filename="a.pdf", data=data)

    assert result.page_count == 2
    assert "First page" in result.text
    assert "Second page" in result.text
    assert result.metadata["extractor"] == "pypdf"
    assert result.metadata["page_count"] == 2


def test_pdf_extractor_reports_per_page_char_counts() -> None:
    data = build_pdf(["abc", "defgh"])
    result = PdfTextExtractor().extract(filename="a.pdf", data=data)

    per_page = result.metadata["per_page_chars"]
    assert len(per_page) == 2
    assert all(count > 0 for count in per_page)


def test_pdf_extractor_rejects_a_non_pdf_payload() -> None:
    with pytest.raises(ExtractionError):
        PdfTextExtractor().extract(filename="a.pdf", data=b"this is definitely not a pdf")


def test_pdf_extractor_rejects_a_blank_document() -> None:
    with pytest.raises(ExtractionError, match="No extractable text"):
        PdfTextExtractor().extract(filename="blank.pdf", data=build_pdf(["   "]))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("a.txt", ".txt"), ("REPORT.PDF", ".pdf"), ("no/extension", ""), ("weird.tar.gz", ".gz")],
)
def test_extension_of_handles_windows_paths_and_case(filename: str, expected: str) -> None:
    assert ExtractorRegistry.extension_of(filename) == expected


@pytest.mark.parametrize("filename", ["a.txt", "b.pdf", "C:\\docs\\c.PDF"])
def test_registry_dispatches_by_extension(filename: str) -> None:
    extractor = ExtractorRegistry().get(filename)
    assert extractor.supports(ExtractorRegistry.extension_of(filename))


def test_registry_rejects_unknown_extension() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        ExtractorRegistry().get("malware.exe")


def test_registry_rejects_a_missing_extension() -> None:
    with pytest.raises(UnsupportedMediaTypeError, match="no extension"):
        ExtractorRegistry().get("README")


def test_registry_lists_supported_extensions() -> None:
    assert ExtractorRegistry().supported_extensions == frozenset({".pdf", ".txt"})
