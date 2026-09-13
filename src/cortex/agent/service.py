"""Agent orchestration service — single-agent + tool-calling loop.

Architecture (Phase 10)
-----------------------
``AgentService.run()`` has four phases, all sharing a single
:class:`~cortex.agent.state.AgentState` object:

1. **Build state** — security gate, load bounded conversation history,
   persist user message, construct :class:`~cortex.agent.state.AgentState`.

2. **Tool-calling loop** — dispatched to one of two strategies:

   * **Native path** (preferred) — used when the ``LLMProvider`` implements
     :class:`~cortex.llm.base.SupportsToolCalling`.  Maintains a typed
     message history and calls
     :meth:`~cortex.llm.base.SupportsToolCalling.generate_with_tools`.

   * **Prompt-based fallback** — used when the provider does *not* implement
     ``SupportsToolCalling``.  Embeds tool schemas in the system prompt and
     parses the LLM's JSON decision.  Identical to the Phase 8 loop.

   Both strategies write their results into ``state.tool_calls`` and
   ``state.retrieved_chunks``.

3. **Grounded final answer** —
   :class:`~cortex.services.prompt_builder.PromptBuilder` constructs a
   grounded prompt from the accumulated retrieval chunks **and the loaded
   conversation history**, then :class:`~cortex.llm.base.LLMProvider`
   generates the final citation-grounded answer.

4. **Persist** — assistant message and token usage are written back to the
   conversation if one was provided.

Memory boundaries
-----------------
* ``state.history``  — previous ``Message`` rows from the DB, bounded by
  ``conversation_history_limit``.  Injected into the **final grounded-answer
  prompt only** (``PromptBuilder``).  Never passed to the tool-calling loop
  to prevent compounding unverified context into tool decisions.
* ``state.retrieved_chunks`` — live RAG results from this run's tool calls.
  Always authoritative; used for citation construction and context grounding.

Provider agnosticism
--------------------
``AgentService`` imports only from :mod:`cortex.agent.types` and
:mod:`cortex.agent.state`, never from the Gemini SDK.

Conversation security
---------------------
When ``conversation_id`` is supplied, :meth:`get_history` enforces ownership
(raises ``ForbiddenError`` / ``NotFoundError``).  History is loaded **before**
any tool call, so an ownership violation aborts the entire run.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING

from cortex.agent.prompt import AgentPromptBuilder
from cortex.agent.registry import ToolRegistry
from cortex.agent.result import AgentResult, ToolCallRecord
from cortex.agent.state import AgentState
from cortex.agent.tools.rag_search import RAGSearchTool
from cortex.agent.types import AgentMessage, ToolCallRequest, ToolResult
from cortex.core.exceptions import BadRequestError, ServiceUnavailableError
from cortex.services.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from cortex.db.models.user import User
    from cortex.llm.base import LLMProvider
    from cortex.services.conversation import ConversationService

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOOL_CALLS = 5
_DEFAULT_HISTORY_LIMIT = 10


class AgentService:
    """Orchestrate a single-agent + tool-calling loop with conversation memory.

    Parameters
    ----------
    llm_provider:
        The application-scoped LLM provider.  If it also implements
        :class:`~cortex.llm.base.SupportsToolCalling`, the native
        function-calling path is used automatically.
    tool_registry:
        Registry pre-populated with the available tools.
    prompt_builder:
        The existing PromptBuilder used for the final grounded-answer step.
    conversation_service:
        Optional — when supplied, the agent enforces conversation ownership,
        loads bounded conversation history, and persists the user/assistant
        message pair.
    max_tool_calls:
        Maximum number of tool invocations per agent run.  Prevents infinite
        loops in case the LLM keeps requesting tools.
    conversation_history_limit:
        Maximum number of previous messages loaded from the database to
        include as conversation memory in the grounded-answer prompt.
        Bounded to prevent unbounded context growth.  Defaults to 10.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        conversation_service: ConversationService | None = None,
        max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
        conversation_history_limit: int = _DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self._llm = llm_provider
        self._registry = tool_registry
        self._prompt_builder = prompt_builder
        self._conv_service = conversation_service
        self._max_tool_calls = max_tool_calls
        self._history_limit = conversation_history_limit
        self._agent_prompt_builder = AgentPromptBuilder()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        question: str,
        user: User,
        conversation_id: str | None = None,
    ) -> AgentResult:
        """Run the agent on a user question and return the grounded answer.

        Parameters
        ----------
        question:
            The user's natural-language question.
        user:
            Authenticated owner — passed through to every tool for ownership
            enforcement.
        conversation_id:
            When set, ownership is verified, bounded conversation history is
            loaded, and the user/assistant messages are persisted on
            completion.

        Returns
        -------
        AgentResult
            Contains the final answer, tool-call trace, and accumulated
            retrieval results for citation construction.
        """
        question = question.strip()
        if not question:
            raise BadRequestError(
                "Agent question must not be empty",
                details={"field": "question"},
            )

        # ------------------------------------------------------------------
        # Phase 1: Build AgentState
        #   a) Load bounded history (ownership enforced inside get_history)
        #   b) Persist user message
        #   c) Construct AgentState with all context for this run
        # ------------------------------------------------------------------
        state = await self._build_state(
            question=question,
            user=user,
            conversation_id=conversation_id,
        )

        # ------------------------------------------------------------------
        # Phase 2: Tool-calling loop — dispatched by provider capability
        # ------------------------------------------------------------------
        from cortex.llm.base import SupportsToolCalling  # local → avoids cycle

        if isinstance(self._llm, SupportsToolCalling):
            logger.debug(
                "Agent using native tool-calling path (provider=%s)",
                type(self._llm).__name__,
            )
            await self._native_tool_loop(state=state)
        else:
            logger.debug(
                "Agent using prompt-based tool-calling path (provider=%s)",
                type(self._llm).__name__,
            )
            await self._prompt_tool_loop(state=state)

        # ------------------------------------------------------------------
        # Phase 3: Generate grounded final answer
        #   Pass conversation history so the LLM can maintain context/tone.
        #   Retrieved chunks remain the authoritative evidence section.
        #
        #   Memory boundary:
        #     state.history  → PromptBuilder history section (context/memory)
        #     state.retrieved_chunks → PromptBuilder context section (evidence)
        # ------------------------------------------------------------------
        grounded_prompt = self._prompt_builder.build(
            question=question,
            retrieved_chunks=state.retrieved_chunks,
            history=state.history if state.has_history else None,
        )

        try:
            final_answer = await self._llm.generate(grounded_prompt)
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            raise ServiceUnavailableError(
                "LLM final-answer generation failed",
                details={"reason": str(exc)},
            ) from exc

        logger.info(
            "Agent run complete: user_id=%s conversation_id=%s "
            "tool_calls=%d chunks=%d history=%d elapsed_ms=%.1f",
            user.id,
            conversation_id,
            state.total_tool_calls,
            len(state.retrieved_chunks),
            len(state.history),
            state.elapsed_ms,
        )

        # ------------------------------------------------------------------
        # Phase 4: Persist assistant message + token usage
        # ------------------------------------------------------------------
        if conversation_id and self._conv_service:
            citation_dicts = [
                {
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                }
                for c in state.retrieved_chunks
            ]
            await self._conv_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=final_answer,
                citations=citation_dicts or None,
            )
            prompt_tokens = len(grounded_prompt) // 4
            completion_tokens = len(final_answer) // 4
            await self._conv_service.record_token_usage(
                conversation_id=conversation_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        return AgentResult(
            answer=final_answer,
            tool_calls=state.tool_calls,
            retrieved_chunks=state.retrieved_chunks,
        )

    # ------------------------------------------------------------------
    # Private — State construction
    # ------------------------------------------------------------------

    async def _build_state(
        self,
        *,
        question: str,
        user: User,
        conversation_id: str | None,
    ) -> AgentState:
        """Build the initial AgentState for this run.

        Loads bounded conversation history (enforcing ownership) and persists
        the user's message before constructing the state object.
        """
        history = []

        if conversation_id and self._conv_service:
            # Load history first (get_history enforces ownership internally)
            history = await self._conv_service.get_history(
                conversation_id=conversation_id,
                user_id=user.id,
                limit=self._history_limit,
            )
            logger.debug(
                "Loaded conversation history: conversation_id=%s messages=%d",
                conversation_id,
                len(history),
            )

            # Persist the incoming user message
            await self._conv_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            await self._conv_service.set_auto_title_if_needed(
                conversation_id=conversation_id,
                first_message=question,
            )

        return AgentState(
            question=question,
            user=user,
            conversation_id=conversation_id,
            history=history,
        )

    # ------------------------------------------------------------------
    # Private — Native tool-calling loop (Phase 9)
    # ------------------------------------------------------------------

    async def _native_tool_loop(self, *, state: AgentState) -> None:
        """Run the tool-calling loop using the provider's native API.

        Writes results into ``state.tool_calls`` and
        ``state.retrieved_chunks`` in-place.
        """
        from cortex.llm.base import SupportsToolCalling

        assert isinstance(self._llm, SupportsToolCalling)  # guaranteed by caller

        # Seed conversation history with the user's question
        messages: list[AgentMessage] = [
            AgentMessage(role="user", text=state.question)
        ]

        for iteration in range(self._max_tool_calls):
            logger.debug(
                "Agent native loop iteration=%d user_id=%s",
                iteration + 1,
                state.user.id,
            )

            try:
                result = await self._llm.generate_with_tools(
                    messages=messages,
                    tool_schemas=self._registry.tool_schemas,
                )
            except ServiceUnavailableError:
                raise
            except Exception as exc:
                raise ServiceUnavailableError(
                    "LLM generate_with_tools failed during agent loop",
                    details={"reason": str(exc)},
                ) from exc

            if result.is_text:
                logger.info(
                    "Agent native loop: text response after %d tool call(s)",
                    state.total_tool_calls,
                )
                break

            # LLM requested a tool call
            tc: ToolCallRequest = result.tool_call  # type: ignore[assignment]
            tool_name = tc.tool_name

            # Append model's tool-call turn to history
            messages.append(AgentMessage(role="model", tool_call=tc))

            # Dispatch the tool
            try:
                observation = await self._registry.dispatch(
                    name=tool_name,
                    args=tc.args,
                    user=state.user,
                )
            except BadRequestError as exc:
                observation = f"Tool error: {exc.message}"
                logger.warning(
                    "Agent native loop: tool %r raised BadRequestError: %s",
                    tool_name,
                    exc.message,
                )

            # Accumulate structured RAG results for citation
            tool_obj = None
            with contextlib.suppress(BadRequestError):
                tool_obj = self._registry.get_tool(tool_name)
            if isinstance(tool_obj, RAGSearchTool):
                state.retrieved_chunks.extend(tool_obj.last_results)

            state.tool_calls.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    args=tc.args,
                    observation=observation,
                )
            )

            # Append tool result to history so the provider can see it
            messages.append(
                AgentMessage(
                    role="tool",
                    tool_result=ToolResult(
                        tool_name=tool_name,
                        output=observation,
                        call_id=tc.call_id,
                    ),
                )
            )

        else:
            logger.warning(
                "Agent native loop: reached max_tool_calls=%d without text response",
                self._max_tool_calls,
            )

    # ------------------------------------------------------------------
    # Private — Prompt-based tool-calling loop (Phase 8 fallback)
    # ------------------------------------------------------------------

    async def _prompt_tool_loop(self, *, state: AgentState) -> None:
        """Prompt-based tool-calling loop (Phase 8 fallback).

        Used when the LLM provider does **not** implement
        :class:`~cortex.llm.base.SupportsToolCalling`.  Embeds tool schemas
        in the system prompt and parses the LLM's JSON decisions.

        Writes results into ``state.tool_calls`` and
        ``state.retrieved_chunks`` in-place.
        """
        current_prompt = self._agent_prompt_builder.build_initial(
            question=state.question,
            tool_schemas=self._registry.tool_schemas,
        )

        loop_count = 0

        while loop_count < self._max_tool_calls:
            loop_count += 1
            logger.debug(
                "Agent prompt loop iteration=%d user_id=%s",
                loop_count,
                state.user.id,
            )

            try:
                raw_response = await self._llm.generate(current_prompt)
            except ServiceUnavailableError:
                raise
            except Exception as exc:
                raise ServiceUnavailableError(
                    "LLM generation failed during agent loop",
                    details={"reason": str(exc)},
                ) from exc

            decision = self._parse_decision(raw_response)
            action = decision.get("action", "")

            if action == "final_answer":
                logger.info(
                    "Agent prompt loop: final_answer after %d tool call(s)",
                    state.total_tool_calls,
                )
                break

            if action == "tool_call":
                tool_name = str(decision.get("tool", "")).strip()
                tool_args: dict = decision.get("args", {})

                try:
                    observation = await self._registry.dispatch(
                        name=tool_name,
                        args=tool_args,
                        user=state.user,
                    )
                except BadRequestError as exc:
                    observation = f"Tool error: {exc.message}"
                    tool_name = tool_name or "unknown"

                tool_obj = None
                with contextlib.suppress(BadRequestError):
                    tool_obj = self._registry.get_tool(tool_name)
                if isinstance(tool_obj, RAGSearchTool):
                    state.retrieved_chunks.extend(tool_obj.last_results)

                state.tool_calls.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        args=tool_args,
                        observation=observation,
                    )
                )

                current_prompt = self._agent_prompt_builder.build_observation_turn(
                    previous_prompt=current_prompt,
                    llm_decision=raw_response,
                    tool_name=tool_name,
                    observation=observation,
                )
            else:
                logger.warning(
                    "Agent prompt loop: unexpected action=%r; treating as final answer",
                    action,
                )
                break

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_decision(raw: str) -> dict:
        """Extract a JSON decision dict from the LLM's raw text response.

        Strips markdown code fences if present.  Falls back to an empty
        dict on parse failure so the loop can exit gracefully.
        """
        text = raw.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first (```json or ```) and last (```) lines
            inner_lines = (
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
            text = "\n".join(inner_lines).strip()

        try:
            decision = json.loads(text)
            if not isinstance(decision, dict):
                raise ValueError("Expected a JSON object")  # noqa: TRY301
            return decision
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Agent could not parse LLM decision as JSON: %s (raw=%r)",
                exc,
                raw[:300],
            )
            # Return a synthetic final-answer so the loop exits cleanly
            return {"action": "final_answer", "answer": raw.strip()}
