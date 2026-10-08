"""Prompt templates for the conversational RAG pipeline.

Two prompts make up the pipeline:

1. :data:`CONDENSE_PROMPT` rewrites a context-dependent follow-up ("and the
   second one?") into a standalone query that can be embedded.
2. :data:`ANSWER_PROMPT` generates the grounded answer from retrieved context
   plus the recent conversation.

Both are LangChain :class:`~langchain_core.prompts.ChatPromptTemplate` objects
and both carry a task tag on their system message (see
:mod:`app.services.llm.tags`) so the offline mock model can behave predictably.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.services.llm.tags import (
    NO_CONTEXT_PLACEHOLDER,
    TASK_TAG_ANSWER,
    TASK_TAG_CONDENSE,
)

#: Instructions for rewriting a follow-up into a standalone question.
CONDENSE_INSTRUCTIONS = """\
Given the conversation so far and a follow-up message, rewrite the follow-up as a
standalone question that makes sense on its own.

Rules:
- Resolve pronouns and implicit references using the conversation.
- Preserve the user's intent and any names, dates or identifiers.
- Do not answer the question.
- If the follow-up is already standalone, return it unchanged.
- Reply with the standalone question only, with no preamble or quotes."""

#: Instructions for answering strictly from retrieved context.
ANSWER_INSTRUCTIONS = """\
You are a document assistant for a company knowledge base. Answer the user's
question using ONLY the retrieved context below.

Rules:
- Ground every claim in the context; never use outside knowledge.
- If the context does not contain the answer, say so plainly and stop.
- Cite the sources you used as [1], [2], ... matching the context numbering.
- Be concise and direct. Prefer the user's terminology.
- If the user's question is a greeting or small talk, respond briefly and
  offer to answer questions about the documents."""

CONDENSE_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", f"{TASK_TAG_CONDENSE}\n{CONDENSE_INSTRUCTIONS}"),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ],
)

ANSWER_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", f"{TASK_TAG_ANSWER}\n{ANSWER_INSTRUCTIONS}"),
        MessagesPlaceholder("history", optional=True),
        (
            "human",
            "Retrieved context:\n<context>\n{context}\n</context>\n\nQuestion: {question}",
        ),
    ],
)

#: Used when retrieval returns nothing, so the model is never handed an empty block.
EMPTY_CONTEXT_PLACEHOLDER: str = NO_CONTEXT_PLACEHOLDER

__all__ = [
    "ANSWER_PROMPT",
    "CONDENSE_PROMPT",
    "EMPTY_CONTEXT_PLACEHOLDER",
]
