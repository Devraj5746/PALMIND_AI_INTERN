"""Factory that builds the configured LangChain chat model."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings
from app.core.enums import LLMProvider
from app.core.errors import ConfigurationError
from app.core.logging import get_logger
from app.services.llm.mock_chat_model import MockChatModel

logger = get_logger(__name__)


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Instantiate the chat model selected by ``LLM_PROVIDER``.

    Returns a :class:`BaseChatModel`, so the rest of the application is written
    against LangChain's interface and never against a specific vendor.

    Raises:
        ConfigurationError: when the selection is unsupported or credentials
            are missing.
    """
    provider = settings.llm_provider

    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        api_key = settings.require_openai_key()
        logger.info("Chat model: OpenAI '%s'", settings.llm_model)
        return ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=api_key,
            openai_api_base=settings.openai_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_output_tokens,
            request_timeout=settings.llm_request_timeout_seconds,
            max_retries=3,
        )

    if provider == LLMProvider.MOCK:
        logger.warning(
            "Using the offline 'mock' chat model: answers are deterministic and "
            "derived from retrieved context only. Set LLM_PROVIDER=openai for real answers.",
        )
        return MockChatModel()

    raise ConfigurationError(  # pragma: no cover - guarded by the enum
        f"Unsupported LLM provider '{provider}'.",
        details={"supported_providers": [p.value for p in LLMProvider]},
    )


def describe_chat_model(model: BaseChatModel) -> str:
    """Return a human-readable identifier for a chat model.

    ``BaseChatModel`` does not standardise a model-name attribute, so this
    probes the common ones and falls back to LangChain's ``_llm_type``.
    """
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return str(model._llm_type)


__all__ = ["build_chat_model", "describe_chat_model"]
