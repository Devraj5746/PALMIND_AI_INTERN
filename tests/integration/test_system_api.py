"""HTTP-level tests for service metadata, health, stats and error handling."""

from __future__ import annotations

from httpx import AsyncClient

from tests.helpers import HANDBOOK_TEXT


async def test_root_reports_service_metadata(client: AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "RAG Backend"
    assert body["version"] == "1.0.0"
    assert body["environment"] == "development"
    assert body["api_prefix"] == "/api/v1"
    assert body["docs"] == "/docs"


async def test_health_reports_every_dependency(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "1.0.0"
    assert body["checked_at"]

    components = {component["name"]: component for component in body["components"]}
    assert set(components) == {
        "database",
        "vector_store",
        "chat_memory",
        "embedding_provider",
        "chat_model",
    }
    assert all(component["healthy"] for component in components.values())
    assert "dim=" in components["embedding_provider"]["detail"]


async def test_stats_describes_an_empty_service(client: AsyncClient) -> None:
    response = await client.get("/api/v1/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == 0
    assert body["chunks"] == 0
    assert body["vectors"] == 0
    assert body["bookings"] == 0
    assert body["embedding_provider"] == "hash"
    assert body["llm_provider"] == "mock"
    assert body["chunking_strategies"] == ["fixed", "recursive"]
    assert body["collection_name"]


async def test_stats_tracks_the_ingested_corpus(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/documents",
        files={"file": ("handbook.txt", HANDBOOK_TEXT.encode(), "text/plain")},
        data={"chunk_size": "300", "chunk_overlap": "40"},
    )
    await client.post(
        "/api/v1/bookings",
        json={
            "name": "Devraj Singh",
            "email": "devraj@example.com",
            "date": "2027-01-05",
            "time": "09:00",
        },
    )

    body = (await client.get("/api/v1/stats")).json()

    assert body["documents"] == 1
    assert body["chunks"] > 0
    assert body["vectors"] == body["chunks"]
    assert body["bookings"] == 1


async def test_openapi_schema_is_served(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "RAG Backend"
    assert "/api/v1/documents" in schema["paths"]
    assert "/api/v1/chat" in schema["paths"]
    assert "/api/v1/bookings" in schema["paths"]
    assert "/api/v1/health" in schema["paths"]


async def test_documentation_endpoints_are_served(client: AsyncClient) -> None:
    assert (await client.get("/docs")).status_code == 200
    assert (await client.get("/redoc")).status_code == 200


async def test_unknown_route_uses_the_standard_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/api/v1/nope")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "http_404"
    assert "message" in error
    assert "details" in error


async def test_wrong_method_uses_the_standard_error_envelope(client: AsyncClient) -> None:
    response = await client.delete("/api/v1/health")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "http_405"


async def test_error_responses_carry_a_request_id(client: AsyncClient) -> None:
    response = await client.get("/api/v1/nope")

    assert response.headers["x-request-id"]


async def test_cors_preflight_is_answered(client: AsyncClient) -> None:
    response = await client.options(
        "/api/v1/chat",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    # ``allow_credentials=True`` means the origin is echoed back rather than
    # answered with a bare ``*``, which browsers reject for credentialed calls.
    assert response.headers["access-control-allow-origin"] == "https://example.com"
    assert response.headers["access-control-allow-credentials"] == "true"
