"""The offline, deterministic LangChain chat model."""

from __future__ import annotations

from datetime import date, time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import Settings
from app.core.enums import EmbeddingProvider, LLMProvider
from app.schemas.booking import ExtractedBookingFields
from app.services.llm.factory import build_chat_model, describe_chat_model
from app.services.llm.mock_chat_model import NO_CONTEXT_ANSWER, MockChatModel
from app.services.rag.prompts import ANSWER_PROMPT, CONDENSE_PROMPT, EMPTY_CONTEXT_PLACEHOLDER

CONTEXT = (
    "[1] source=handbook.txt chunk=0 score=0.912\n"
    "Every full-time employee is entitled to twenty four days of paid annual leave."
)


@pytest.fixture
def model() -> MockChatModel:
    return MockChatModel(reference_date=date(2026, 1, 15))


def test_model_is_a_langchain_chat_model(model: MockChatModel) -> None:
    from langchain_core.language_models.chat_models import BaseChatModel

    assert isinstance(model, BaseChatModel)
    assert model._llm_type == "mock-deterministic"


async def test_answer_step_uses_the_retrieved_context(model: MockChatModel) -> None:
    messages = ANSWER_PROMPT.format_messages(
        context=CONTEXT,
        question="How many days of annual leave?",
        history=[],
    )
    response = await model.ainvoke(messages)

    assert "Based on the retrieved context" in response.content
    assert "twenty four days" in response.content
    assert "[1]" in response.content


async def test_answer_step_declines_when_no_context_was_retrieved(model: MockChatModel) -> None:
    messages = ANSWER_PROMPT.format_messages(
        context=EMPTY_CONTEXT_PLACEHOLDER,
        question="What is the leave policy?",
        history=[],
    )
    response = await model.ainvoke(messages)

    assert NO_CONTEXT_ANSWER in response.content


async def test_condense_step_returns_the_standalone_question(model: MockChatModel) -> None:
    messages = CONDENSE_PROMPT.format_messages(
        history=[HumanMessage(content="How many days of leave?"), AIMessage(content="24 days.")],
        question="and the second one?",
    )
    response = await model.ainvoke(messages)

    assert response.content.strip() == "and the second one?"


async def test_answer_step_ignores_history_and_uses_only_the_latest_turn(
    model: MockChatModel,
) -> None:
    messages = ANSWER_PROMPT.format_messages(
        context=CONTEXT,
        question="Summarise that.",
        history=[HumanMessage(content="earlier question"), AIMessage(content="earlier answer")],
    )
    response = await model.ainvoke(messages)

    assert "twenty four days" in response.content


async def test_untagged_prompts_are_echoed_back(model: MockChatModel) -> None:
    response = await model.ainvoke([HumanMessage(content="hello")])
    assert response.content == "[mock-llm] hello"


async def test_answer_prefix_is_configurable() -> None:
    model = MockChatModel(answer_prefix="[offline]")
    response = await model.ainvoke([HumanMessage(content="hi")])
    assert response.content == "[offline] hi"


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


async def test_structured_output_returns_typed_fields(model: MockChatModel) -> None:
    chain = model.with_structured_output(ExtractedBookingFields)
    result = await chain.ainvoke(
        [HumanMessage(content="I'm Devraj Singh, devraj@example.com, 2026-12-20 at 09:30")],
    )

    assert isinstance(result, ExtractedBookingFields)
    assert result.name == "Devraj Singh"
    assert result.email == "devraj@example.com"
    assert result.date == date(2026, 12, 20)
    assert result.time == time(9, 30)
    assert result.missing_fields == []


async def test_structured_output_uses_the_reference_date_for_relative_dates(
    model: MockChatModel,
) -> None:
    chain = model.with_structured_output(ExtractedBookingFields)
    result = await chain.ainvoke([HumanMessage(content="book for tomorrow at 10:00")])

    assert result.date == date(2026, 1, 16)


async def test_structured_output_can_include_the_raw_message(model: MockChatModel) -> None:
    chain = model.with_structured_output(ExtractedBookingFields, include_raw=True)
    result = await chain.ainvoke([HumanMessage(content="devraj@example.com")])

    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert isinstance(result["raw"], AIMessage)
    assert result["parsed"].email == "devraj@example.com"
    assert result["parsing_error"] is None


async def test_structured_output_accepts_a_plain_dict_schema(model: MockChatModel) -> None:
    chain = model.with_structured_output({"title": "fields"})
    result = await chain.ainvoke([HumanMessage(content="2026-12-20")])

    assert result["date"] == date(2026, 12, 20)


async def test_structured_output_ignores_prompt_history(model: MockChatModel) -> None:
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            ("system", "extract"),
            MessagesPlaceholder("history", optional=True),
            ("human", "{message}"),
        ],
    )
    messages = prompt.format_messages(
        message="09:30",
        history=[HumanMessage(content="Devraj Singh")],
    )
    result = await model.with_structured_output(ExtractedBookingFields).ainvoke(messages)

    assert result.time == time(9, 30)
    # The name came from history, not the latest turn, so it must not leak in.
    assert result.name is None


async def test_structured_output_handles_a_bare_name_message(model: MockChatModel) -> None:
    result = await model.with_structured_output(ExtractedBookingFields).ainvoke(
        [HumanMessage(content="Devraj Singh")],
    )
    assert result.name == "Devraj Singh"
    assert result.missing_fields == ["email", "date", "time"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_builds_the_mock_model() -> None:
    model = build_chat_model(Settings(_env_file=None, llm_provider=LLMProvider.MOCK))

    assert isinstance(model, MockChatModel)
    assert describe_chat_model(model) == "mock-deterministic"


def test_factory_returns_a_langchain_model_for_openai_selection() -> None:
    model = build_chat_model(
        Settings(
            _env_file=None,
            llm_provider=LLMProvider.OPENAI,
            embedding_provider=EmbeddingProvider.HASH,
            openai_api_key="sk-test",
            llm_model="gpt-4o-mini",
        ),
    )
    # No network call is made at construction time.
    assert describe_chat_model(model) == "gpt-4o-mini"
