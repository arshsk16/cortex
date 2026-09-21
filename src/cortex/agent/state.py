"""AgentState - centralized carrier for a single agent run''s memory context.

Phase 10 introduces a clean ``AgentState`` dataclass that replaces the loose
local variables previously scattered through
:meth:`~cortex.agent.service.AgentService.run`.

Phase 14 adds:

* ``plan`` - an optional ordered list of tool names the planner suggests
  calling before the loop starts (prompt-based path only).
* ``budget_exhausted`` - set to ``True`` when the tool-call budget is
  consumed without a ``final_answer`` decision, so the final answer step
  can communicate the constraint to the LLM.
* ``tool_call_summary()`` - compact text rendering of the tool trace,
  injected into the grounded-answer prompt for better synthesis quality.

Design principles
-----------------
* **Immutable construction** - built once from loaded history + current question;
  the ``tool_calls`` and ``retrieved_chunks`` fields are mutable lists that grow
  during the tool-calling loop, but the identity of the state object is fixed.
* **Explicit memory boundary** - ``history`` (conversation memory) and
  ``retrieved_chunks`` (document grounding) are kept as separate, named fields.
  They must never be mixed: history provides context/tone; retrieved chunks are
  the authoritative evidence the LLM must ground its final answer in.
* **Provider-agnostic** - contains no Gemini SDK types; carries only
  :mod:`cortex.agent.types` and :mod:`cortex.agent.result` objects.
* **Extensible** - fields are dataclass fields; future phases (semantic memory,
  session embeddings, user preferences) can add new fields without changing
  existing code paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cortex.agent.result import ToolCallRecord
from cortex.retrieval.models import RetrievalResult
from cortex.schemas.memory import MemorySearchResult

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
    run_id:
        Unique UUID identifying this specific agent execution. Used for
        ephemeral state indexing in Redis (Phase 12).
    started_at_iso:
        ISO 8601 UTC timestamp when this run was initiated.
    history:
        Previous ``Message`` rows loaded from the database, in chronological
        order, bounded by ``conversation_history_limit``.  Used **only** for
        the final grounded-answer prompt; never passed to the tool-calling loop.
        Empty list for stateless runs or first messages.
    tool_calls:
        Ordered trace of every tool invocation in this run.  Grows during the
        tool-calling loop.  Each record includes status and duration_ms
        (Phase 14).
    retrieved_chunks:
        All :class:`~cortex.retrieval.models.RetrievalResult` objects
        accumulated across all RAG search tool calls.  Used for citation
        construction and grounded-answer prompt construction.
    memory_hits:
        Long-term memory entries retrieved at run start (Phase 13B).
    plan:
        Ordered list of tool names the planner suggests calling (Phase 14).
        Empty list when planning is disabled or the native tool-calling path
        is used (the provider plans internally).
    budget_exhausted:
        Set to ``True`` when the tool-call budget (``max_tool_calls``) is
        consumed without the LLM returning a ``final_answer`` decision
        (Phase 14).  The final answer step uses this to instruct the LLM to
        synthesise a best-effort answer from what it has.
    started_at:
        ``time.perf_counter()`` value captured at the start of ``run()``.
        Used to compute total execution time in the log.
    extra:
        Reserved dict for future extensions (e.g. user preferences, session
        metadata).  Not used in Phase 10-14.
    """

    question: str
    user: User
    conversation_id: str | None

    # Run identity and timing (Phase 12)
    run_id: str = field(default_factory=lambda: str(uuid4()))
    started_at_iso: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    # Conversation memory (bounded by history_limit)
    history: list[Message] = field(default_factory=list)

    # In-flight tool state (grows during tool-calling loop)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)

    # Long-term semantic memory hits retrieved at run start (Phase 13B)
    memory_hits: list[MemorySearchResult] = field(default_factory=list)

    # Phase 14: Planning and budget tracking
    plan: list[str] = field(default_factory=list)
    budget_exhausted: bool = False

    # Execution metadata
    started_at: float = field(default_factory=time.perf_counter)

    # Extensibility hook
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_memory_hits(self) -> bool:
        """True when long-term memory entries were retrieved for this run."""
        return bool(self.memory_hits)

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

    def tool_call_summary(self) -> str:
        """Return a compact multi-line summary of all tool calls made.

        Used to inject the tool trace into the grounded-answer prompt so
        the LLM has full context of what was retrieved/computed.

        Returns an empty string when no tools were called.

        Format (one line per call)::

            [rag_search] query="Paris" -> Paris is the capital... (ok, 123ms)
            [calculator] expression="2+2" -> Result: 4 (ok, 0ms)
        """
        if not self.tool_calls:
            return ""

        lines: list[str] = []
        for tc in self.tool_calls:
            # Truncate args values for readability
            args_preview = ", ".join(
                f'{k}="{str(v)[:60]}"' for k, v in tc.args.items()
            )
            obs_preview = tc.observation[:200].replace("\n", " ")
            if len(tc.observation) > 200:
                obs_preview += "..."
            lines.append(
                f"[{tc.tool_name}] {args_preview} -> {obs_preview}"
                f" ({tc.status}, {tc.duration_ms:.0f}ms)"
            )
        return "\n".join(lines)
