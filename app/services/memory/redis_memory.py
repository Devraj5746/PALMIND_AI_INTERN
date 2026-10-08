"""Redis-backed conversation memory and booking drafts.

Storage layout (one key per concern, all namespaced and TTL'd):

* ``chat:history:<session_id>``     — a Redis list of JSON-encoded messages,
  oldest first, trimmed to the configured sliding window.
* ``chat:booking_draft:<session_id>`` — a JSON object holding the partially
  filled booking, discarded as soon as the booking is created or cancelled.

A single connection pool is shared by every session; ``ChatMemoryStore`` is the
owner of that pool and hands out lightweight per-session views.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from redis.asyncio import Redis

from app.core.config import Settings
from app.core.constants import REDIS_BOOKING_DRAFT_PREFIX, REDIS_CHAT_HISTORY_PREFIX
from app.core.errors import DependencyUnavailableError
from app.core.logging import get_logger
from app.schemas.booking import BookingDraft
from app.services.memory.base import BookingDraftStore, ChatMemory

logger = get_logger(__name__)

#: Named alias for the client type this module uses.  Every client is created with
#: ``decode_responses=True``, so payloads come back as ``str`` rather than bytes.
RedisClient = Redis


def build_redis_client(settings: Settings) -> RedisClient:
    """Create an async Redis client from settings.

    ``decode_responses=True`` means values come back as ``str``; all payloads
    here are JSON, so that is what we want.
    """
    return Redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5.0,
        socket_timeout=5.0,
        health_check_interval=30,
    )


def _serialise(message: BaseMessage) -> str:
    return json.dumps(message_to_dict(message), default=str)


def _deserialise(raw: str | bytes) -> BaseMessage:
    # ``json.loads`` accepts both ``str`` and ``bytes``, which keeps this correct
    # whether or not the client was created with ``decode_responses=True``.
    return messages_from_dict([json.loads(raw)])[0]


class RedisChatMemory(ChatMemory):
    """A single session's transcript, stored as a Redis list."""

    def __init__(
        self,
        *,
        client: RedisClient,
        session_id: str,
        max_messages: int,
        ttl_seconds: int,
    ) -> None:
        super().__init__(
            session_id=session_id,
            max_messages=max_messages,
            ttl_seconds=ttl_seconds,
        )
        self._client = client
        self._key = f"{REDIS_CHAT_HISTORY_PREFIX}:{session_id}"

    async def get_messages(self) -> list[BaseMessage]:
        try:
            raw = await self._client.lrange(self._key, 0, -1)
        except Exception as exc:
            raise self._unavailable(exc, "reading history") from exc

        messages: list[BaseMessage] = []
        for item in raw:
            try:
                messages.append(_deserialise(item))
            except (ValueError, KeyError, TypeError):
                # A single corrupt entry must not take down the conversation.
                logger.warning("Skipping malformed history entry for session %s", self.session_id)
        return messages

    async def append(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return

        payload = [_serialise(message) for message in messages]
        try:
            pipeline = self._client.pipeline()
            pipeline.rpush(self._key, *payload)
            # Keep only the newest N messages (sliding window).
            pipeline.ltrim(self._key, -self.max_messages, -1)
            pipeline.expire(self._key, self.ttl_seconds)
            await pipeline.execute()
        except Exception as exc:
            raise self._unavailable(exc, "appending history") from exc

    async def clear(self) -> int:
        try:
            pipeline = self._client.pipeline()
            pipeline.llen(self._key)
            pipeline.delete(self._key)
            results = await pipeline.execute()
        except Exception as exc:
            raise self._unavailable(exc, "clearing history") from exc
        return int(results[0] or 0)

    async def health(self) -> None:
        try:
            await self._client.ping()
        except Exception as exc:
            raise self._unavailable(exc, "health check") from exc

    @staticmethod
    def _unavailable(exc: Exception, operation: str) -> DependencyUnavailableError:
        logger.error("Redis chat-memory operation failed while %s: %s", operation, exc)
        return DependencyUnavailableError(
            "Chat memory is unavailable.",
            details={"operation": operation, "reason": str(exc)},
        )


class RedisBookingDraftStore(BookingDraftStore):
    """Stores partial bookings as JSON keys in Redis."""

    def __init__(self, *, client: RedisClient, ttl_seconds: int) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{REDIS_BOOKING_DRAFT_PREFIX}:{session_id}"

    async def get(self, session_id: str) -> BookingDraft | None:
        try:
            raw = await self._client.get(self._key(session_id))
        except Exception as exc:
            raise self._unavailable(exc, "reading a booking draft") from exc

        if not raw:
            return None
        try:
            return BookingDraft.model_validate_json(raw)
        except ValueError:
            logger.warning("Discarding malformed booking draft for session %s", session_id)
            return None

    async def save(self, session_id: str, draft: BookingDraft) -> None:
        try:
            await self._client.set(
                self._key(session_id),
                draft.model_dump_json(),
                ex=self._ttl_seconds,
            )
        except Exception as exc:
            raise self._unavailable(exc, "saving a booking draft") from exc

    async def clear(self, session_id: str) -> None:
        try:
            await self._client.delete(self._key(session_id))
        except Exception as exc:
            raise self._unavailable(exc, "clearing a booking draft") from exc

    @staticmethod
    def _unavailable(exc: Exception, operation: str) -> DependencyUnavailableError:
        logger.error("Redis booking-draft operation failed while %s: %s", operation, exc)
        return DependencyUnavailableError(
            "Booking storage is unavailable.",
            details={"operation": operation, "reason": str(exc)},
        )


class ChatMemoryStore:
    """Owns the Redis client and creates per-session memory views."""

    def __init__(self, client: RedisClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._drafts = RedisBookingDraftStore(
            client=client,
            ttl_seconds=settings.chat_memory_ttl_seconds,
        )

    @property
    def client(self) -> RedisClient:
        return self._client

    @property
    def drafts(self) -> RedisBookingDraftStore:
        return self._drafts

    def for_session(self, session_id: str) -> RedisChatMemory:
        """Return the chat memory view for ``session_id``."""
        return RedisChatMemory(
            client=self._client,
            session_id=session_id,
            max_messages=self._settings.chat_memory_max_messages,
            ttl_seconds=self._settings.chat_memory_ttl_seconds,
        )

    async def clear_session(self, session_id: str) -> int:
        """Delete both the transcript and any draft for a session."""
        deleted = await self.for_session(session_id).clear()
        await self._drafts.clear(session_id)
        return deleted

    async def health(self) -> None:
        try:
            await self._client.ping()
        except Exception as exc:
            raise DependencyUnavailableError(
                "Chat memory is unavailable.",
                details={"operation": "health check", "reason": str(exc)},
            ) from exc

    async def close(self) -> None:
        """Release the connection pool."""
        try:
            await self._client.aclose()
        except Exception:
            logger.debug("Failed to close the Redis client cleanly", exc_info=True)


__all__ = [
    "ChatMemoryStore",
    "RedisBookingDraftStore",
    "RedisChatMemory",
    "RedisClient",
    "build_redis_client",
]
