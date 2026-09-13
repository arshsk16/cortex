"""AgentState — centralized carrier for a single agent run's memory context.

Phase 10 introduces a clean ``AgentState`` dataclass that replaces the loose
local variables previously scattered through
:meth:`~cortex.agent.service.AgentService.run`.

Design principles
-----------------
* **Immutable construction** — built once from loaded history + current question;
  the ``tool_calls`` and ``retrieved_chunks`` fields are mutable lists that grow
  during the tool-calling loop, but the identity of the state object is fixed.
* **Explicit memory boundary** — ``history`` (conversation memory) and
  ``retrieved_chunks`` (document grounding) are kept as separate, named fields.
  They must never be mixed: history provides context/tone; retrieved chunks are
  the authoritative evidence the LLM must ground its final answer in.
* **Provider-agnostic** — contains no Gemini SDK types; carries only
  :mod:`cortex.agent.types` and :mod:`cortex.agent.result` objects.
* **Extensible** — fields are dataclass fields; future phases (semantic memory,
  session embeddings, user preferences) can add new fields without changing
  existing code paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cortex.agent.result import ToolCallRecord
from cortex.retrieval.models import RetrievalResult

if TYPE_CHECKING:
    from cortex.db.models.conversation import Message
    from cortex.db.models.user import User


@dataclass
class AgentState:
    """Carries all in-flight state for a single agent run.

    Attributes
    ----------
    question:
        The current user question (stripped).
    user:
        Authenticated owner of the run.  Passed through to every tool for
        ownership enforcement; never serialised or logged in full.
    conversation_id:
        The conversation this run belongs to, or ``None`` for stateless runs.
    history:
        Previous ``Message`` rows loaded from the database, in chronological
        order, bounded by ``conversation_history_limit``.  Used **only** for
        the final grounded-answer prompt; never passed to the tool-calling loop.
        Empty list for stateless runs or first messages.
    tool_calls:
        Ordered trace of every tool invocation in this run.  Grows during the
        tool-calling loop.
    retrieved_chunks:
        All :class:`~cortex.retrieval.models.RetrievalResult` objects
        accumulated across all RAG search tool calls.  Used for citation
        construction and grounded-answer prompt construction.
    started_at:
        ``time.perf_counter()`` value captured at the start of ``run()``.
        Used to compute total execution time in the log.
    extra:
        Reserved dict for future extensions (e.g. semantic memory hits,
        user preferences, session metadata).  Not used in Phase 10.
    """

    question: str
    user: User
    conversation_id: str | None

    # Conversation memory (bounded by history_limit)
    history: list[Message] = field(default_factory=list)

    # In-flight tool state (grows during tool-calling loop)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)

    # Execution metadata
    started_at: float = field(default_factory=time.perf_counter)

    # Extensibility hook — unused in Phase 10
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since this AgentState was created."""
        return (time.perf_counter() - self.started_at) * 1000

    @property
    def has_history(self) -> bool:
        """True when previous conversation messages are available."""
        return bool(self.history)

    @property
    def total_tool_calls(self) -> int:
        """Number of tool invocations made so far in this run."""
        return len(self.tool_calls)
