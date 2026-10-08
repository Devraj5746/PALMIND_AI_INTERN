"""The custom conversational RAG engine.

This is the piece the task explicitly asks to be hand-written: **no
``RetrievalQAChain``**, no ``create_retrieval_chain``, no hidden LCEL chain that
obscures the retrieval step.  The pipeline is four explicit, observable steps:

1. **Condense** — if there is prior conversation, rewrite the follow-up into a
   standalone query (pronoun resolution) so retrieval is not blinded by context.
2. **Retrieve** — embed the standalone query and pull the nearest chunks from
   Qdrant (always visible as :attr:`RAGAnswer.sources`).
3. **Assemble** — render a numbered context block and build the chat prompt with
   the recent conversation.
4. **Generate** — call the chat model and return the answer with citations.

Every step is logged and every intermediate value is returned to the caller, so
the behaviour is inspectable rather than buried inside a framework abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.rag.prompts import (
    ANSWER_PROMPT,
    CONDENSE_PROMPT,
    EMPTY_CONTEXT_PLACEHOLDER,
)
from app.services.rag.retriever import ChunkRetriever
from app.services.vector_store.base import RetrievedChunk

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """The result of one RAG turn."""

    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    standalone_query: str = ""
    used_history: bool = False
    context_chars: int = 0


class ConversationalRAGEngine:
    """Grounded, multi-turn question answering over the ingested corpus."""

    def __init__(
        self,
        *,
        chat_model: BaseChatModel,
        retriever: ChunkRetriever,
        settings: Settings,
    ) -> None:
        self._chat_model = chat_model
        self._retriever = retriever
        self._settings = settings

    # -- public API --------------------------------------------------------
    async def answer(
        self,
        question: str,
        *,
        history: list[BaseMessage] | None = None,
        top_k: int | None = None,
        document_id: UUID | str | None = None,
        use_retrieval: bool = True,
    ) -> RAGAnswer:
        """Answer ``question``, optionally grounded in retrieved context."""
        history = history or []
        window = self._history_window(history)

        # 1. Condense ------------------------------------------------------
        standalone_query = await self._condense(question, window)

        # 2. Retrieve ------------------------------------------------------
        chunks: list[RetrievedChunk] = []
        if use_retrieval:
            chunks = await self._retriever.retrieve(
                standalone_query,
                top_k=top_k,
                document_id=document_id,
            )

        # 3. Assemble ------------------------------------------------------
        context = self._retriever.format_context(chunks) or EMPTY_CONTEXT_PLACEHOLDER
        messages = ANSWER_PROMPT.format_messages(
            context=context,
            question=question,
            history=window,
        )

        # 4. Generate ------------------------------------------------------
        logger.debug(
            "Generating answer (standalone_query=%r, chunks=%d, history=%d)",
            standalone_query[:80],
            len(chunks),
            len(window),
        )
        response = await self._chat_model.ainvoke(messages)
        answer_text = self._content_to_text(response.content)

        return RAGAnswer(
            answer=answer_text,
            sources=chunks,
            standalone_query=standalone_query,
            used_history=bool(window),
            context_chars=len(context),
        )

    # -- steps -------------------------------------------------------------
    async def _condense(self, question: str, history: list[BaseMessage]) -> str:
        """Rewrite a follow-up into a standalone query when history exists."""
        if not history or not self._settings.rag_condense_followups:
            return question

        messages = CONDENSE_PROMPT.format_messages(history=history, question=question)
        try:
            response = await self._chat_model.ainvoke(messages)
        except Exception:
            logger.warning(
                "Query condensing failed; falling back to the raw question", exc_info=True
            )
            return question

        rewritten = self._content_to_text(response.content).strip()
        return rewritten or question

    def _history_window(self, history: list[BaseMessage]) -> list[BaseMessage]:
        """Cap the number of prior turns handed to the model."""
        limit = self._settings.chat_memory_max_messages
        return history[-limit:] if len(history) > limit else list(history)

    @staticmethod
    def _content_to_text(content: str | list[str | dict[str, Any]]) -> str:
        """Normalise a message content payload (str, or list of content blocks)."""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)


__all__ = ["ConversationalRAGEngine", "RAGAnswer"]
