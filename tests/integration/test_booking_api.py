"""HTTP-level tests for the REST interview-booking endpoints."""

from __future__ import annotations

import datetime as dt
import uuid

from httpx import AsyncClient

BOOKINGS = "/api/v1/bookings"


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "name": "Devraj Singh",
        "email": "devraj@example.com",
        "date": (dt.date.today() + dt.timedelta(days=14)).isoformat(),
        "time": "14:30",
        "notes": None,
    }
    body.update(overrides)
    return body


async def test_create_booking(client: AsyncClient) -> None:
    response = await client.post(BOOKINGS, json=payload())

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Devraj Singh"
    assert body["email"] == "devraj@example.com"
    assert body["time"] == "14:30:00"
    assert body["source"] == "api"
    assert body["session_id"] is None
    assert uuid.UUID(body["id"])


async def test_email_is_normalised_to_lower_case(client: AsyncClient) -> None:
    response = await client.post(BOOKINGS, json=payload(email="Devraj@Example.COM"))
    assert response.json()["email"] == "devraj@example.com"


async def test_notes_are_persisted(client: AsyncClient) -> None:
    response = await client.post(BOOKINGS, json=payload(notes="Needs a quiet room"))
    assert response.json()["notes"] == "Needs a quiet room"


async def test_duplicate_slot_for_the_same_candidate_conflicts(client: AsyncClient) -> None:
    assert (await client.post(BOOKINGS, json=payload())).status_code == 201

    response = await client.post(BOOKINGS, json=payload())

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "conflict"
    assert "already exists" in error["message"]


async def test_the_same_slot_is_allowed_for_a_different_candidate(client: AsyncClient) -> None:
    await client.post(BOOKINGS, json=payload())

    response = await client.post(BOOKINGS, json=payload(email="priya@example.com", name="Priya"))

    assert response.status_code == 201


async def test_a_different_slot_is_allowed_for_the_same_candidate(client: AsyncClient) -> None:
    await client.post(BOOKINGS, json=payload())

    response = await client.post(BOOKINGS, json=payload(time="16:00"))

    assert response.status_code == 201


async def test_create_booking_rejects_a_past_date(client: AsyncClient) -> None:
    past = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    response = await client.post(BOOKINGS, json=payload(date=past))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_create_booking_rejects_an_invalid_email(client: AsyncClient) -> None:
    response = await client.post(BOOKINGS, json=payload(email="not-an-email"))

    assert response.status_code == 422
    assert "email" in str(response.json()["error"]["details"])


async def test_create_booking_rejects_a_blank_name(client: AsyncClient) -> None:
    response = await client.post(BOOKINGS, json=payload(name="   "))
    assert response.status_code == 422


async def test_create_booking_rejects_a_malformed_time(client: AsyncClient) -> None:
    response = await client.post(BOOKINGS, json=payload(time="half past two"))
    assert response.status_code == 422


async def test_create_booking_requires_every_field(client: AsyncClient) -> None:
    response = await client.post(BOOKINGS, json={"name": "Devraj Singh"})
    assert response.status_code == 422


async def test_list_bookings(client: AsyncClient) -> None:
    await client.post(BOOKINGS, json=payload(email="a@example.com", name="A"))
    await client.post(BOOKINGS, json=payload(email="b@example.com", name="B"))

    response = await client.get(BOOKINGS)

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2


async def test_list_bookings_can_filter_by_email(client: AsyncClient) -> None:
    await client.post(BOOKINGS, json=payload(email="a@example.com", name="A"))
    await client.post(BOOKINGS, json=payload(email="b@example.com", name="B"))

    body = (await client.get(BOOKINGS, params={"email": "a@example.com"})).json()

    assert body["total"] == 1
    assert body["items"][0]["email"] == "a@example.com"


async def test_list_bookings_supports_pagination(client: AsyncClient) -> None:
    for index in range(3):
        await client.post(
            BOOKINGS,
            json=payload(email=f"user{index}@example.com", name=f"User {index}"),
        )

    body = (await client.get(BOOKINGS, params={"limit": 2, "offset": 1})).json()

    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["offset"] == 1


async def test_list_bookings_is_empty_to_begin_with(client: AsyncClient) -> None:
    body = (await client.get(BOOKINGS)).json()
    assert body["total"] == 0
    assert body["items"] == []


async def test_get_booking_by_id(client: AsyncClient) -> None:
    created = (await client.post(BOOKINGS, json=payload())).json()

    response = await client.get(f"{BOOKINGS}/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


async def test_get_booking_returns_404_for_an_unknown_id(client: AsyncClient) -> None:
    response = await client.get(f"{BOOKINGS}/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_delete_booking(client: AsyncClient) -> None:
    created = (await client.post(BOOKINGS, json=payload())).json()

    deleted = await client.delete(f"{BOOKINGS}/{created['id']}")

    assert deleted.status_code == 204
    assert (await client.get(f"{BOOKINGS}/{created['id']}")).status_code == 404
    assert (await client.get(BOOKINGS)).json()["total"] == 0


async def test_deleting_a_booking_frees_the_slot(client: AsyncClient) -> None:
    created = (await client.post(BOOKINGS, json=payload())).json()
    await client.delete(f"{BOOKINGS}/{created['id']}")

    response = await client.post(BOOKINGS, json=payload())

    assert response.status_code == 201


async def test_delete_booking_returns_404_for_an_unknown_id(client: AsyncClient) -> None:
    response = await client.delete(f"{BOOKINGS}/{uuid.uuid4()}")
    assert response.status_code == 404
