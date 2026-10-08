"""A deterministic, offline LangChain chat model.

Purpose: let the entire stack run — and be tested — with **no API key and no
network**, while still exercising the real control flow (prompt templates,
message history, structured output, the RAG loop).

It is a genuine :class:`~langchain_core.language_models.chat_models.BaseChatModel`,
so it is a drop-in replacement for ``ChatOpenAI`` everywhere in the codebase:

* :meth:`_generate` answers by reading the task tag on the system prompt
  (see :mod:`app.services.llm.tags`) — it echoes the follow-up for a condense
  step, and answers from the ``<context>`` block for a RAG step.
* :meth:`with_structured_output` delegates to the rule-based extractor in
  :mod:`app.services.llm.rules`.

It is explicitly *not* a language model: it has no world knowledge and never
invents facts.  That is what makes it useful as a test double.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from app.services.llm import rules
from app.services.llm.tags import (
    CONTEXT_CLOSE_TAG,
    CONTEXT_OPEN_TAG,
    NO_CONTEXT_PLACEHOLDER,
    TASK_TAG_ANSWER,
    TASK_TAG_CONDENSE,
)

#: Maximum number of context characters echoed back in a mock answer.
MAX_ECHO_CHARS: int = 400
NO_CONTEXT_ANSWER: str = "I don't have enough information in the provided documents to answer that."


def _coerce_messages(value: Any) -> list[BaseMessage]:
    """Normalise a runnable input into a list of chat messages."""
    if isinstance(value, list):
        return [
            item if isinstance(item, BaseMessage) else HumanMessage(content=str(item))
            for item in value
        ]
    if isinstance(value, BaseMessage):
        return [value]
    if hasattr(value, "to_messages"):
        return list(value.to_messages())
    return [HumanMessage(content=str(value))]


def _system_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages if m.type == "system")


def _last_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.type == "human":
            return str(message.content)
    return ""


def _between(text: str, open_tag: str, close_tag: str) -> str | None:
    start = text.find(open_tag)
    if start == -1:
        return None
    end = text.find(close_tag, start + len(open_tag))
    if end == -1:
        return None
    return text[start + len(open_tag) : end]


class MockChatModel(BaseChatModel):
    """Deterministic stand-in for a hosted chat model."""

    # -- configurable behaviour (pydantic fields) --------------------------
    answer_prefix: str = "[mock-llm]"
    reference_date: date | None = None
    """Fixed 'today' used by date parsing, so tests are time-independent."""

    @property
    def _llm_type(self) -> str:
        return "mock-deterministic"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"answer_prefix": self.answer_prefix, "reference_date": self.reference_date}

    # -- generation --------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._respond(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _respond(self, messages: list[BaseMessage]) -> str:
        system = _system_text(messages)
        last_human = _last_human_text(messages)

        if TASK_TAG_CONDENSE in system:
            # A condense step asks for a standalone question: the follow-up.
            return last_human.strip()

        if TASK_TAG_ANSWER in system:
            context = _between(last_human, CONTEXT_OPEN_TAG, CONTEXT_CLOSE_TAG)
            stripped = context.strip() if context else ""
            if stripped and stripped != NO_CONTEXT_PLACEHOLDER:
                snippet = " ".join(stripped.split())[:MAX_ECHO_CHARS]
                return f"{self.answer_prefix} Based on the retrieved context: {snippet}"
            return f"{self.answer_prefix} {NO_CONTEXT_ANSWER}"

        return f"{self.answer_prefix} {last_human.strip()}"

    # -- structured output -------------------------------------------------
    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        """Return a runnable that fills ``schema`` from the latest user message."""

        def _extract(value: Any, config: Any = None, **_: Any) -> Any:
            messages = _coerce_messages(value)
            payload = rules.parse_booking_fields(
                _last_human_text(messages),
                today=self.reference_date,
            )
            parsed = self._validate(schema, payload)
            if not include_raw:
                return parsed
            return {
                "raw": AIMessage(content=json.dumps(payload, default=str)),
                "parsed": parsed,
                "parsing_error": None,
            }

        return RunnableLambda(_extract)

    @staticmethod
    def _validate(schema: dict[str, Any] | type, payload: dict[str, Any]) -> Any:
        """Coerce the parsed payload into ``schema``, ignoring unknown keys."""
        if isinstance(schema, dict):
            return payload

        # Callers pass a Pydantic model class when they want typed output.
        model_cls = cast("type[BaseModel]", schema)
        declared = set(model_cls.model_fields)
        filtered = {key: value for key, value in payload.items() if key in declared}
        return model_cls.model_validate(filtered)


__all__ = ["MockChatModel"]
