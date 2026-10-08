"""Task tags embedded in prompt system messages.

The offline :class:`~app.services.llm.mock_chat_model.MockChatModel` has to know
what a prompt is asking for without a language model to reason about it.  Rather
than guessing from prose, every system prompt in
:mod:`app.services.rag.prompts` carries an explicit task tag.

Real providers simply ignore the tag — it is plain text in the system message —
so there is no behavioural difference in production.  Keeping the tags in a
dedicated module avoids a circular import between the prompts and the mock.
"""

from __future__ import annotations

from typing import Final

TASK_TAG_CONDENSE: Final[str] = "[[task:condense]]"
TASK_TAG_ANSWER: Final[str] = "[[task:answer]]"
TASK_TAG_CONVERSE: Final[str] = "[[task:converse]]"

#: Delimiters used by the RAG prompt to fence retrieved chunks.
CONTEXT_OPEN_TAG: Final[str] = "<context>"
CONTEXT_CLOSE_TAG: Final[str] = "</context>"

#: Text substituted when retrieval returns nothing.  Shared here so that both
#: the prompt template and the offline model agree on its meaning.
NO_CONTEXT_PLACEHOLDER: Final[str] = "(no relevant context was retrieved)"
