"""Test helpers: synthetic documents and small assertion utilities.

Everything here is deterministic and dependency-free so the suite stays
hermetic — no fixture files, no network, no clock dependence.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

from app.schemas.ingestion import IngestionOptions

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from app.container import Container
    from app.services.ingestion_service import IngestionResult

# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------

#: A small "company handbook" with several distinct paragraphs.  Deliberately
#: written so that the recursive chunker has real structure to work with and the
#: lexical (hash) embeddings have distinctive vocabulary to match on.
HANDBOOK_TEXT: Final[str] = """\
Acme Corp Employee Handbook

Annual leave

Every full-time employee is entitled to twenty four days of paid annual leave \
per calendar year. Leave accrues monthly and unused days may be carried over \
into the first quarter of the following year.

Sick leave

Absences longer than three consecutive days require a medical certificate. \
Short absences should be reported to your manager before the start of the \
working day.

Remote work

Employees may work remotely up to three days per week. Remote days must be \
recorded in the scheduling tool so that the team knows where everyone is.

Equipment

The company provides a laptop and a monitor. Replacement equipment is ordered \
through the IT help desk and normally arrives within five working days.
"""

#: Vocabulary that only appears in one paragraph, used to assert that retrieval
#: actually discriminates between chunks.
LEAVE_QUERY: Final[str] = "How many days of paid annual leave do I get?"
REMOTE_QUERY: Final[str] = "How many days can I work remotely each week?"
EQUIPMENT_QUERY: Final[str] = "How do I order replacement equipment?"


def build_pdf(page_texts: list[str]) -> bytes:
    """Build a minimal, valid single-font PDF containing ``page_texts``.

    ``pypdf`` is a reader, not a writer, and the project deliberately avoids
    pulling in a PDF-authoring dependency just for tests.  Emitting the PDF
    syntax directly (with a correct cross-reference table) is short, exact and
    keeps the suite free of extra packages.
    """
    if not page_texts:
        raise ValueError("at least one page is required")

    font_id = 3
    objects: dict[int, bytes] = {}
    page_ids: list[int] = []

    for index, text in enumerate(page_texts):
        content_id = 4 + 2 * index
        page_id = 5 + 2 * index
        page_ids.append(page_id)

        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        objects[content_id] = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>".encode("latin-1")
    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    buffer = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    highest = max(objects)

    for object_id in range(1, highest + 1):
        if object_id not in objects:
            continue
        offsets[object_id] = len(buffer)
        buffer += f"{object_id} 0 obj\n".encode() + objects[object_id] + b"\nendobj\n"

    xref_offset = len(buffer)
    buffer += f"xref\n0 {highest + 1}\n".encode()
    buffer += b"0000000000 65535 f \n"
    for object_id in range(1, highest + 1):
        if object_id in offsets:
            buffer += f"{offsets[object_id]:010d} 00000 n \n".encode()
        else:
            buffer += b"0000000000 65535 f \n"
    buffer += (
        f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return bytes(buffer)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


# ---------------------------------------------------------------------------
# Ingestion helper
# ---------------------------------------------------------------------------


async def ingest_text(
    container: Container,
    text: str,
    *,
    filename: str = "handbook.txt",
    content_type: str = "text/plain",
    chunk_size: int = 300,
    chunk_overlap: int = 40,
) -> IngestionResult:
    """Ingest ``text`` through the real ingestion pipeline (no HTTP)."""
    return await container.ingestion_service.ingest(
        filename=filename,
        content_type=content_type,
        data=text.encode("utf-8"),
        options=IngestionOptions(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    )


__all__ = [
    "EQUIPMENT_QUERY",
    "HANDBOOK_TEXT",
    "LEAVE_QUERY",
    "REMOTE_QUERY",
    "build_pdf",
    "cosine_similarity",
    "ingest_text",
]
