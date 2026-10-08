"""HTTP-level tests for the conversational RAG API (RAG + interview booking)."""

from __future__ import annotations

import datetime as dt

from httpx import AsyncClient

from tests.helpers import HANDBOOK_TEXT

CHAT = "/api/v1/chat"
DOCUMENTS = "/api/v1/documents"


def future_date(days: int = 30) -> str:
    return (dt.date.today() + dt.timedelta(days=days)).isoformat()


async def seed_documents(client: AsyncClient) -> str:
    response = await client.post(
        DOCUMENTS,
        files={"file": ("handbook.txt", HANDBOOK_TEXT.encode(), "text/plain")},
        data={"chunk_size": "300", "chunk_overlap": "40"},
    )
    assert response.status_code == 201
    return response.json()["id"]


# ---------------------------------------------------------------------------
# Retrieval-grounded turns
# ---------------------------------------------------------------------------


async def test_chat_answers_from_the_ingested_documents(client: AsyncClient) -> None:
    await seed_documents(client)

    response = await client.post(CHAT, json={"message": "How many days of annual leave?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "rag"
    assert body["session_id"]
    assert body["sources"], "expected the answer to cite retrieved chunks"
    assert body["model"]
    assert "annual leave" in body["answer"].lower()
    assert body["sources"][0]["snippet"]


async def test_citations_are_truncated_snippets_not_whole_chunks(client: AsyncClient) -> None:
    await seed_documents(client)

    body = (await client.post(CHAT, json={"message": "Remote work policy?"})).json()

    # noinspection PyTypeChecker
    for source in body["sources"]:
        assert len(source["snippet"]) <= 280
        assert source["chunk_index"] >= 0
        assert source["score"] >= 0


async def test_a_new_session_id_is_returned_when_none_was_supplied(client: AsyncClient) -> None:
    first = (await client.post(CHAT, json={"message": "hello"})).json()
    second = (await client.post(CHAT, json={"message": "hello"})).json()

    assert first["session_id"]
    assert second["session_id"]
    assert first["session_id"] != second["session_id"]


async def test_multi_turn_conversation_reuses_the_session(client: AsyncClient) -> None:
    await seed_documents(client)
    session = (await client.post(CHAT, json={"message": "How many days of annual leave?"})).json()[
        "session_id"
    ]

    follow_up = await client.post(
        CHAT,
        json={"message": "and what about remote work?", "session_id": session},
    )

    body = follow_up.json()
    assert body["session_id"] == session
    assert body["used_history"] is True
    assert body["standalone_query"]


async def test_retrieval_can_be_disabled_for_a_turn(client: AsyncClient) -> None:
    await seed_documents(client)

    body = (await client.post(CHAT, json={"message": "hello", "use_retrieval": False})).json()

    assert body["sources"] == []
    assert body["route"] == "rag"


async def test_answering_without_any_documents_declines(client: AsyncClient) -> None:
    body = (await client.post(CHAT, json={"message": "What is the leave policy?"})).json()

    assert body["sources"] == []
    assert "don't have enough information" in body["answer"]


async def test_top_k_can_be_overridden_per_turn(client: AsyncClient) -> None:
    await seed_documents(client)

    body = (await client.post(CHAT, json={"message": "leave", "top_k": 1})).json()

    assert len(body["sources"]) <= 1


async def test_chat_rejects_an_empty_message(client: AsyncClient) -> None:
    response = await client.post(CHAT, json={"message": ""})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


async def test_history_records_both_sides_of_every_turn(client: AsyncClient) -> None:
    await seed_documents(client)
    session = (await client.post(CHAT, json={"message": "How many days of leave?"})).json()[
        "session_id"
    ]
    await client.post(CHAT, json={"message": "thanks", "session_id": session})

    response = await client.get(f"{CHAT}/{session}/history")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session
    assert len(body["messages"]) == 4
    assert [message["role"] for message in body["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert body["messages"][0]["content"] == "How many days of leave?"


async def test_history_for_an_unknown_session_is_empty(client: AsyncClient) -> None:
    body = (await client.get(f"{CHAT}/no-such-session/history")).json()

    assert body["messages"] == []


async def test_clearing_a_session_deletes_its_transcript(client: AsyncClient) -> None:
    session = (await client.post(CHAT, json={"message": "hello"})).json()["session_id"]

    cleared = await client.delete(f"{CHAT}/{session}")

    assert cleared.status_code == 200
    assert cleared.json() == {"session_id": session, "deleted_messages": 2}
    assert (await client.get(f"{CHAT}/{session}/history")).json()["messages"] == []


async def test_sessions_do_not_share_memory(client: AsyncClient) -> None:
    first = (await client.post(CHAT, json={"message": "first session"})).json()["session_id"]
    second = (await client.post(CHAT, json={"message": "second session"})).json()["session_id"]

    first_history = (await client.get(f"{CHAT}/{first}/history")).json()["messages"]
    second_history = (await client.get(f"{CHAT}/{second}/history")).json()["messages"]

    assert first_history[0]["content"] == "first session"
    assert second_history[0]["content"] == "second session"


# ---------------------------------------------------------------------------
# Interview booking through the conversation
# ---------------------------------------------------------------------------


async def test_booking_is_completed_across_several_turns(client: AsyncClient) -> None:
    date = future_date()
    session = None
    turns = [
        "I want to book an interview",
        "Devraj Singh",
        "devraj@example.com",
        date,
        "09:30",
    ]

    responses = []
    for message in turns:
        payload: dict[str, object] = {"message": message}
        if session:
            payload["session_id"] = session
        response = await client.post(CHAT, json=payload)
        assert response.status_code == 200
        body = response.json()
        session = body["session_id"]
        responses.append(body)

    # The first turn enters the booking flow but has nothing to book yet.
    assert all(body["route"] == "booking" for body in responses)
    assert all(body["booking"] is None for body in responses[:-1])

    final = responses[-1]
    assert final["booking"] is not None
    assert final["booking"]["name"] == "Devraj Singh"
    assert final["booking"]["email"] == "devraj@example.com"
    assert final["booking"]["date"] == date
    assert final["booking"]["source"] == "chat"
    assert final["booking"]["session_id"] == session
    assert "Booked" in final["answer"]


async def test_the_booking_flow_asks_for_the_fields_it_still_needs(client: AsyncClient) -> None:
    first = (await client.post(CHAT, json={"message": "I would like to book an interview"})).json()

    assert first["booking"] is None
    assert "full name" in first["answer"]

    second = (
        await client.post(
            CHAT,
            json={"message": "Devraj Singh", "session_id": first["session_id"]},
        )
    ).json()

    assert "email address" in second["answer"]


async def test_the_conversation_falls_back_to_rag_mid_booking(client: AsyncClient) -> None:
    await seed_documents(client)
    started = (await client.post(CHAT, json={"message": "I want to book an interview"})).json()
    session = started["session_id"]

    question = await client.post(
        CHAT,
        json={
            "message": "Actually, how many days of annual leave do I get?",
            "session_id": session,
        },
    )

    body = question.json()
    assert body["route"] == "rag"
    assert body["sources"]


async def test_a_booking_created_in_chat_appears_in_the_rest_listing(client: AsyncClient) -> None:
    date = future_date(45)
    session = None
    for message in ["book an interview", "Priya Nair", "priya@example.com", date, "11:00"]:
        payload: dict[str, object] = {"message": message}
        if session:
            payload["session_id"] = session
        body = (await client.post(CHAT, json=payload)).json()
        session = body["session_id"]

    assert body["booking"] is not None

    listing = (await client.get("/api/v1/bookings", params={"email": "priya@example.com"})).json()
    assert listing["total"] == 1
    assert listing["items"][0]["source"] == "chat"


async def test_a_past_date_in_the_booking_flow_is_rejected_politely(client: AsyncClient) -> None:
    session = (await client.post(CHAT, json={"message": "book an interview"})).json()["session_id"]

    response = await client.post(
        CHAT,
        json={"message": "2020-01-01", "session_id": session},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["booking"] is None
    assert "past" in body["answer"].lower()


async def test_cancelling_a_booking_flow_discards_the_draft(client: AsyncClient) -> None:
    session = (await client.post(CHAT, json={"message": "book an interview"})).json()["session_id"]
    await client.post(CHAT, json={"message": "Devraj Singh", "session_id": session})

    # Intent classification is keyword-based, so the phrasing must name what is
    # being cancelled (as the extraction prompt instructs the hosted model to).
    cancelled = (
        await client.post(
            CHAT,
            json={"message": "I want to cancel my interview", "session_id": session},
        )
    ).json()

    assert cancelled["booking"] is None
    assert "cancel" in cancelled["answer"].lower()

    # The draft is gone, so a plain follow-up no longer continues the flow.
    follow_up = (await client.post(CHAT, json={"message": "hello", "session_id": session})).json()
    assert follow_up["route"] == "rag"


async def test_listing_bookings_through_the_conversation(client: AsyncClient) -> None:
    date = future_date(60)
    await client.post(
        "/api/v1/bookings",
        json={"name": "Devraj Singh", "email": "devraj@example.com", "date": date, "time": "15:00"},
    )

    body = (
        await client.post(
            CHAT,
            json={"message": "show my interviews for devraj@example.com"},
        )
    ).json()

    assert body["route"] == "booking"
    assert date in body["answer"]


async def test_a_duplicate_chat_booking_asks_for_another_slot(client: AsyncClient) -> None:
    date = future_date(75)
    await client.post(
        "/api/v1/bookings",
        json={"name": "Devraj Singh", "email": "devraj@example.com", "date": date, "time": "10:00"},
    )

    session = None
    replies = []
    for message in ["book an interview", "Devraj Singh", "devraj@example.com", date, "10:00"]:
        payload: dict[str, object] = {"message": message}
        if session:
            payload["session_id"] = session
        body = (await client.post(CHAT, json=payload)).json()
        session = body["session_id"]
        replies.append(body)

    assert replies[-1]["booking"] is None
    assert "already taken" in replies[-1]["answer"]
