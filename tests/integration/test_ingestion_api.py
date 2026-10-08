"""HTTP-level tests for the document ingestion and retrieval endpoints."""

from __future__ import annotations

import uuid

from httpx import AsyncClient

from tests.helpers import HANDBOOK_TEXT, build_pdf

DOCUMENTS = "/api/v1/documents"


async def upload(
    client: AsyncClient,
    *,
    filename: str = "handbook.txt",
    data: bytes | None = None,
    content_type: str = "text/plain",
    **form: str,
):
    return await client.post(
        DOCUMENTS,
        files={
            "file": (filename, data if data is not None else HANDBOOK_TEXT.encode(), content_type)
        },
        data=form or None,
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


async def test_upload_ingests_a_text_document(client: AsyncClient) -> None:
    response = await upload(
        client, chunking_strategy="recursive", chunk_size="300", chunk_overlap="40"
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "handbook.txt"
    assert body["status"] == "indexed"
    assert body["chunk_count"] > 0
    assert body["chunking_strategy"] == "recursive"
    assert body["chunk_size"] == 300
    assert body["chunk_overlap"] == 40
    assert body["embedding_provider"] == "hash"
    assert body["duration_ms"] >= 0
    assert body["size_kb"] > 0


async def test_upload_ingests_a_pdf_document(client: AsyncClient) -> None:
    pdf = build_pdf(["Annual leave is twenty four days.", "Remote work is three days per week."])

    response = await upload(client, filename="policy.pdf", data=pdf, content_type="application/pdf")

    assert response.status_code == 201
    body = response.json()
    assert body["page_count"] == 2
    assert body["status"] == "indexed"
    assert body["content_type"] == "application/pdf"


async def test_both_chunking_strategies_are_selectable(client: AsyncClient) -> None:
    fixed = await upload(client, chunking_strategy="fixed", chunk_size="200", chunk_overlap="50")
    recursive = await upload(
        client,
        chunking_strategy="recursive",
        chunk_size="200",
        chunk_overlap="50",
    )

    assert fixed.status_code == 201
    assert recursive.status_code == 201
    assert fixed.json()["chunking_strategy"] == "fixed"
    assert recursive.json()["chunking_strategy"] == "recursive"
    # Fixed windows are uniform, so they produce at least as many chunks.
    assert fixed.json()["chunk_count"] >= recursive.json()["chunk_count"]


async def test_upload_without_parameters_uses_the_server_defaults(client: AsyncClient) -> None:
    response = await upload(client)

    assert response.status_code == 201
    body = response.json()
    assert body["chunking_strategy"] == "recursive"
    assert body["chunk_size"] == 1000
    assert body["chunk_overlap"] == 150


async def test_upload_reports_an_unsupported_file_type(client: AsyncClient) -> None:
    response = await upload(client, filename="malware.exe", data=b"MZ\x90\x00")

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


async def test_upload_reports_an_empty_file(client: AsyncClient) -> None:
    response = await upload(client, data=b"")

    assert response.status_code == 415


async def test_upload_reports_an_unknown_chunking_strategy(client: AsyncClient) -> None:
    response = await upload(client, chunking_strategy="semantic")

    assert response.status_code == 422


async def test_upload_rejects_overlap_larger_than_the_chunk_size(client: AsyncClient) -> None:
    response = await upload(client, chunk_size="100", chunk_overlap="500")

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert "chunk_overlap" in str(error["details"])


async def test_upload_rejects_an_out_of_range_chunk_size(client: AsyncClient) -> None:
    response = await upload(client, chunk_size="5")

    assert response.status_code == 422


async def test_upload_requires_a_file(client: AsyncClient) -> None:
    response = await client.post(DOCUMENTS)

    assert response.status_code == 422


async def test_upload_returns_a_request_id_header(client: AsyncClient) -> None:
    response = await upload(client)
    assert response.headers["x-request-id"]
    assert float(response.headers["x-process-time-ms"]) >= 0


async def test_a_supplied_request_id_is_echoed_back(client: AsyncClient) -> None:
    response = await client.get("/", headers={"X-Request-ID": "trace-me-123"})
    assert response.headers["x-request-id"] == "trace-me-123"


# ---------------------------------------------------------------------------
# Listing and detail
# ---------------------------------------------------------------------------


async def test_listing_documents_returns_ingested_ones(client: AsyncClient) -> None:
    first = await upload(client, filename="a.txt")
    await upload(client, filename="b.txt")

    response = await client.get(DOCUMENTS)

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert {item["filename"] for item in body["items"]} == {"a.txt", "b.txt"}
    assert any(item["id"] == first.json()["id"] for item in body["items"])


async def test_listing_supports_pagination(client: AsyncClient) -> None:
    for index in range(3):
        await upload(client, filename=f"file-{index}.txt")

    page = await client.get(DOCUMENTS, params={"limit": 2, "offset": 0})

    body = page.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2


async def test_listing_can_filter_by_status(client: AsyncClient) -> None:
    await upload(client)

    indexed = await client.get(DOCUMENTS, params={"status": "indexed"})
    failed = await client.get(DOCUMENTS, params={"status": "failed"})

    assert indexed.json()["total"] == 1
    assert failed.json()["total"] == 0


async def test_listing_rejects_an_unknown_status_filter(client: AsyncClient) -> None:
    response = await client.get(DOCUMENTS, params={"status": "exploded"})
    assert response.status_code == 422


async def test_document_detail_includes_chunks_with_offsets(client: AsyncClient) -> None:
    uploaded = await upload(client, chunk_size="300", chunk_overlap="40")
    document_id = uploaded.json()["id"]

    response = await client.get(f"{DOCUMENTS}/{document_id}")

    assert response.status_code == 200
    body = response.json()
    assert len(body["chunks"]) == body["chunk_count"]
    first = body["chunks"][0]
    assert first["start_char"] == 0
    assert first["end_char"] == len(first["text"])
    assert first["char_count"] == len(first["text"])
    assert first["token_estimate"] >= 1
    assert first["vector_id"] == first["id"]
    assert first["chunking_strategy"] == "recursive"


async def test_document_detail_returns_404_for_an_unknown_id(client: AsyncClient) -> None:
    response = await client.get(f"{DOCUMENTS}/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_document_detail_rejects_a_malformed_id(client: AsyncClient) -> None:
    response = await client.get(f"{DOCUMENTS}/not-a-uuid")

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def test_search_returns_ranked_hits(client: AsyncClient) -> None:
    await upload(client, chunk_size="300", chunk_overlap="40")

    response = await client.post(
        f"{DOCUMENTS}/search", json={"query": "annual leave days", "top_k": 3}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "annual leave days"
    assert body["results"]
    assert body["results"][0]["score"] >= body["results"][-1]["score"]
    assert body["results"][0]["filename"] == "handbook.txt"


async def test_search_can_be_scoped_to_one_document(client: AsyncClient) -> None:
    target = (await upload(client, filename="handbook.txt")).json()["id"]
    await upload(
        client,
        filename="other.txt",
        data=b"Parking permits are issued by the facilities team.",
    )

    response = await client.post(
        f"{DOCUMENTS}/search",
        json={"query": "annual leave and parking permits", "top_k": 10, "document_id": target},
    )

    results = response.json()["results"]
    assert results
    assert {hit["document_id"] for hit in results} == {target}


async def test_search_on_an_empty_corpus_returns_no_hits(client: AsyncClient) -> None:
    response = await client.post(f"{DOCUMENTS}/search", json={"query": "anything at all"})

    assert response.status_code == 200
    assert response.json()["results"] == []


async def test_search_rejects_an_empty_query(client: AsyncClient) -> None:
    response = await client.post(f"{DOCUMENTS}/search", json={"query": ""})
    assert response.status_code == 422


async def test_search_rejects_an_out_of_range_top_k(client: AsyncClient) -> None:
    response = await client.post(f"{DOCUMENTS}/search", json={"query": "x", "top_k": 500})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def test_delete_removes_the_document_and_its_vectors(client: AsyncClient) -> None:
    document_id = (await upload(client)).json()["id"]

    deleted = await client.delete(f"{DOCUMENTS}/{document_id}")

    assert deleted.status_code == 204
    assert (await client.get(f"{DOCUMENTS}/{document_id}")).status_code == 404
    assert (await client.get(DOCUMENTS)).json()["total"] == 0
    assert (await client.get("/api/v1/stats")).json()["vectors"] == 0


async def test_delete_returns_404_for_an_unknown_id(client: AsyncClient) -> None:
    response = await client.delete(f"{DOCUMENTS}/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_re_ingesting_the_same_content_creates_a_second_document(client: AsyncClient) -> None:
    first = await upload(client, filename="dup.txt")
    second = await upload(client, filename="dup.txt")

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
    assert (await client.get(DOCUMENTS)).json()["total"] == 2
