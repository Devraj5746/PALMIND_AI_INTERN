"""Redis-backed conversation memory and booking-draft storage."""

from __future__ import annotations

from datetime import date, time

import pytest
from fakeredis.aioredis import FakeRedis
from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import Settings
from app.core.constants import REDIS_BOOKING_DRAFT_PREFIX, REDIS_CHAT_HISTORY_PREFIX
from app.schemas.booking import BookingDraft
from app.services.memory.redis_memory import ChatMemoryStore

SESSION_A = "session-a"
SESSION_B = "session-b"


@pytest.fixture
def memory(redis_client: FakeRedis, settings: Settings) -> ChatMemoryStore:
    return ChatMemoryStore(redis_client, settings)


@pytest.fixture
def short_window(redis_client: FakeRedis, settings: Settings) -> ChatMemoryStore:
    """A store that keeps only the last six messages."""
    return ChatMemoryStore(
        redis_client,
        settings.model_copy(update={"chat_memory_max_messages": 6, "chat_memory_ttl_seconds": 60}),
    )


def history_key(session_id: str) -> str:
    return f"{REDIS_CHAT_HISTORY_PREFIX}:{session_id}"


def draft_key(session_id: str) -> str:
    return f"{REDIS_BOOKING_DRAFT_PREFIX}:{session_id}"


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


async def test_a_fresh_session_has_no_history(memory: ChatMemoryStore) -> None:
    assert await memory.for_session(SESSION_A).get_messages() == []


async def test_appending_a_turn_round_trips_roles_and_content(memory: ChatMemoryStore) -> None:
    session = memory.for_session(SESSION_A)

    await session.append_turn(user_message="How much leave do I get?", assistant_message="24 days.")

    messages = await session.get_messages()
    assert [type(message) for message in messages] == [HumanMessage, AIMessage]
    assert messages[0].content == "How much leave do I get?"
    assert messages[1].content == "24 days."


async def test_history_is_ordered_oldest_first(memory: ChatMemoryStore) -> None:
    session = memory.for_session(SESSION_A)
    await session.append_turn(user_message="first", assistant_message="one")
    await session.append_turn(user_message="second", assistant_message="two")

    contents = [message.content for message in await session.get_messages()]
    assert contents == ["first", "one", "second", "two"]


async def test_appending_nothing_is_a_no_op(memory: ChatMemoryStore) -> None:
    session = memory.for_session(SESSION_A)
    await session.append([])
    assert await session.get_messages() == []


async def test_sliding_window_keeps_only_the_newest_messages(short_window: ChatMemoryStore) -> None:
    session = short_window.for_session(SESSION_A)
    for index in range(10):
        await session.append_turn(user_message=f"q{index}", assistant_message=f"a{index}")

    messages = await session.get_messages()
    assert len(messages) == 6
    assert [message.content for message in messages] == ["q7", "a7", "q8", "a8", "q9", "a9"]


async def test_sessions_are_isolated(memory: ChatMemoryStore) -> None:
    await memory.for_session(SESSION_A).append_turn(user_message="for A", assistant_message="ack A")
    await memory.for_session(SESSION_B).append_turn(user_message="for B", assistant_message="ack B")

    a_contents = [message.content for message in await memory.for_session(SESSION_A).get_messages()]
    b_contents = [message.content for message in await memory.for_session(SESSION_B).get_messages()]

    assert a_contents == ["for A", "ack A"]
    assert b_contents == ["for B", "ack B"]


async def test_clear_reports_how_many_messages_were_removed(memory: ChatMemoryStore) -> None:
    session = memory.for_session(SESSION_A)
    await session.append_turn(user_message="one", assistant_message="two")

    assert await session.clear() == 2
    assert await session.get_messages() == []


async def test_clear_on_an_empty_session_returns_zero(memory: ChatMemoryStore) -> None:
    assert await memory.for_session(SESSION_A).clear() == 0


async def test_corrupt_entries_are_skipped_without_losing_the_rest(
    memory: ChatMemoryStore,
    redis_client: FakeRedis,
) -> None:
    session = memory.for_session(SESSION_A)
    await session.append_turn(user_message="good question", assistant_message="good answer")
    # Simulate a partially written or legacy entry.
    await redis_client.rpush(history_key(SESSION_A), "not json at all")

    messages = await session.get_messages()

    assert [message.content for message in messages] == ["good question", "good answer"]


async def test_transcript_sets_a_ttl(
    short_window: ChatMemoryStore, redis_client: FakeRedis
) -> None:
    await short_window.for_session(SESSION_A).append_turn(user_message="q", assistant_message="a")

    ttl = await redis_client.ttl(history_key(SESSION_A))
    assert 0 < ttl <= 60


async def test_health_check_passes_against_the_fake_server(memory: ChatMemoryStore) -> None:
    await memory.health()
    await memory.for_session(SESSION_A).health()


# ---------------------------------------------------------------------------
# Booking drafts
# ---------------------------------------------------------------------------


async def test_a_session_starts_without_a_draft(memory: ChatMemoryStore) -> None:
    assert await memory.drafts.get(SESSION_A) is None


async def test_drafts_round_trip(memory: ChatMemoryStore) -> None:
    draft = BookingDraft(name="Devraj Singh", email="devraj@example.com")

    await memory.drafts.save(SESSION_A, draft)

    restored = await memory.drafts.get(SESSION_A)
    assert restored == draft


async def test_drafts_round_trip_date_and_time_values(memory: ChatMemoryStore) -> None:
    draft = BookingDraft(
        name="Devraj Singh",
        email="devraj@example.com",
        date=date(2027, 1, 5),
        time=time(9, 30),
    )

    await memory.drafts.save(SESSION_A, draft)
    restored = await memory.drafts.get(SESSION_A)

    assert restored is not None
    assert restored.date == date(2027, 1, 5)
    assert restored.time == time(9, 30)
    assert restored.is_complete


async def test_drafts_are_isolated_per_session(memory: ChatMemoryStore) -> None:
    await memory.drafts.save(SESSION_A, BookingDraft(name="A"))
    await memory.drafts.save(SESSION_B, BookingDraft(name="B"))

    assert (await memory.drafts.get(SESSION_A)).name == "A"  # type: ignore[union-attr]
    assert (await memory.drafts.get(SESSION_B)).name == "B"  # type: ignore[union-attr]


async def test_drafts_can_be_cleared(memory: ChatMemoryStore) -> None:
    await memory.drafts.save(SESSION_A, BookingDraft(name="Devraj"))
    await memory.drafts.clear(SESSION_A)
    assert await memory.drafts.get(SESSION_A) is None


async def test_malformed_drafts_are_discarded_rather_than_raised(
    memory: ChatMemoryStore,
    redis_client: FakeRedis,
) -> None:
    await redis_client.set(draft_key(SESSION_A), "{not valid json")

    assert await memory.drafts.get(SESSION_A) is None


async def test_drafts_set_a_ttl(short_window: ChatMemoryStore, redis_client: FakeRedis) -> None:
    await short_window.drafts.save(SESSION_A, BookingDraft(name="Devraj"))

    ttl = await redis_client.ttl(draft_key(SESSION_A))
    assert 0 < ttl <= 60


async def test_clear_session_removes_history_and_draft(memory: ChatMemoryStore) -> None:
    session = memory.for_session(SESSION_A)
    await session.append_turn(user_message="q", assistant_message="a")
    await memory.drafts.save(SESSION_A, BookingDraft(name="Devraj"))

    deleted = await memory.clear_session(SESSION_A)

    assert deleted == 2
    assert await session.get_messages() == []
    assert await memory.drafts.get(SESSION_A) is None


async def test_clear_session_leaves_other_sessions_untouched(memory: ChatMemoryStore) -> None:
    await memory.for_session(SESSION_B).append_turn(user_message="keep", assistant_message="me")

    await memory.clear_session(SESSION_A)

    assert len(await memory.for_session(SESSION_B).get_messages()) == 2


async def test_close_releases_the_client(memory: ChatMemoryStore) -> None:
    await memory.close()


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [("chat_memory_max_messages", 0), ("chat_memory_ttl_seconds", 0)],
)
async def test_invalid_window_configuration_is_rejected(
    redis_client: FakeRedis,
    settings: Settings,
    field: str,
    value: int,
) -> None:
    store = ChatMemoryStore(redis_client, settings.model_copy(update={field: value}))
    with pytest.raises(ValueError):
        store.for_session(SESSION_A)
