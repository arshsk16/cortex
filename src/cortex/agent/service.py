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

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from cortex.agent.events import AgentEvent
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
    from cortex.state_store.base import StateStore

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOOL_CALLS = 5
_DEFAULT_HISTORY_LIMIT = 10
_DEFAULT_STATE_TTL = 1800  # 30 minutes


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
    state_store:
        Optional ephemeral state store (Redis or Null).  When supplied,
        agent execution snapshots are saved at run start, after each tool
        call, and deleted on successful completion (or marked ``failed``
        on error).  Defaults to ``NullStateStore`` behaviour (no-op).
    state_ttl_seconds:
        TTL for Redis state snapshots.  Defaults to 1800 (30 minutes).
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        conversation_service: ConversationService | None = None,
        state_store: StateStore | None = None,
        max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
        conversation_history_limit: int = _DEFAULT_HISTORY_LIMIT,
        state_ttl_seconds: int = _DEFAULT_STATE_TTL,
    ) -> None:
        self._llm = llm_provider
        self._registry = tool_registry
        self._prompt_builder = prompt_builder
        self._conv_service = conversation_service
        self._state_store = state_store
        self._max_tool_calls = max_tool_calls
        self._history_limit = conversation_history_limit
        self._state_ttl = state_ttl_seconds
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

        try:
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

            # Clean up ephemeral state on success
            await self._delete_state(state)

            return AgentResult(
                answer=final_answer,
                tool_calls=state.tool_calls,
                retrieved_chunks=state.retrieved_chunks,
            )
        except Exception as exc:
            await self._save_state(state, status="failed", error=str(exc))
            raise

    async def stream(
        self,
        *,
        question: str,
        user: User,
        conversation_id: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent and yield SSE events incrementally.

        Yields
        ------
        ToolCallEvent
            Immediately when the LLM selects a tool (before execution).
        ToolResultEvent
            After the tool returns (observation truncated to 500 chars).
        TokenEvent
            Each text chunk of the final streamed LLM answer.
        DoneEvent
            On successful completion; mirrors AgentResponse field names.
        ErrorEvent
            On unrecoverable failure; stream ends.

        The tool-calling loop (native or prompt-based) runs identically to
        :meth:`run` — not streamed.  Only the final grounded-answer LLM call
        uses :meth:`~cortex.llm.base.LLMProvider.generate_stream`.

        Conversation ownership, history loading, user-message persistence, and
        assistant-message persistence follow the same rules as :meth:`run`.
        """
        from cortex.agent.events import (
            DoneEvent,
            ErrorEvent,
            TokenEvent,
        )
        from cortex.llm.base import SupportsToolCalling

        question = question.strip()
        if not question:
            yield ErrorEvent(message="Agent question must not be empty")
            return

        # ------------------------------------------------------------------
        # Phase 1: Build AgentState (security gate + history + persist user msg)
        # ------------------------------------------------------------------
        try:
            state = await self._build_state(
                question=question,
                user=user,
                conversation_id=conversation_id,
            )
        except Exception as exc:
            yield ErrorEvent(message=str(exc))
            return

        try:
            # ------------------------------------------------------------------
            # Phase 2: Tool-calling loop with event emission
            # ------------------------------------------------------------------
            if isinstance(self._llm, SupportsToolCalling):
                async for event in self._native_tool_loop_stream(state=state):
                    yield event
            else:
                async for event in self._prompt_tool_loop_stream(state=state):
                    yield event

            # ------------------------------------------------------------------
            # Phase 3: Stream final grounded answer token-by-token
            # ------------------------------------------------------------------
            grounded_prompt = self._prompt_builder.build(
                question=question,
                retrieved_chunks=state.retrieved_chunks,
                history=state.history if state.has_history else None,
            )

            final_parts: list[str] = []
            try:
                token_stream = await self._llm.generate_stream(grounded_prompt)
                async for token in token_stream:
                    if token:
                        final_parts.append(token)
                        yield TokenEvent(text=token)
            except ServiceUnavailableError as exc:
                await self._save_state(state, status="failed", error=str(exc))
                yield ErrorEvent(message=f"LLM streaming failed: {exc}")
                return
            except asyncio.CancelledError:
                await self._save_state(
                    state, status="cancelled", error="Client disconnected"
                )
                raise
            except Exception as exc:
                await self._save_state(state, status="failed", error=str(exc))
                yield ErrorEvent(message=f"LLM streaming failed: {exc}")
                return

            final_answer = "".join(final_parts)

            logger.info(
                "Agent stream complete: user_id=%s conversation_id=%s "
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

            # Clean up ephemeral state on success
            await self._delete_state(state)

            # Build done event (mirrors AgentResponse field names)
            tool_calls_made = [
                {
                    "tool_name": tc.tool_name,
                    "args": tc.args,
                    "observation": tc.observation,
                }
                for tc in state.tool_calls
            ]
            citations = [
                {
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                }
                for c in state.retrieved_chunks
            ]
            yield DoneEvent(
                answer=final_answer,
                tool_calls_made=tool_calls_made,
                citations=citations,
                conversation_id=conversation_id,
            )
        except asyncio.CancelledError:
            await self._save_state(
                state, status="cancelled", error="Client disconnected"
            )
            raise
        except Exception as exc:
            await self._save_state(state, status="failed", error=str(exc))
            yield ErrorEvent(message=f"Tool loop failed: {exc}")
            return

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

        state = AgentState(
            question=question,
            user=user,
            conversation_id=conversation_id,
            history=history,
        )
        await self._save_state(state, status="in_progress")
        return state

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
            await self._save_state(state, status="in_progress")

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
                await self._save_state(state, status="in_progress")

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
    # Private — Streaming tool-calling loops (Phase 11)
    # ------------------------------------------------------------------

    async def _native_tool_loop_stream(
        self, *, state: AgentState
    ) -> AsyncGenerator[AgentEvent, None]:
        """Native tool-calling loop that yields ToolCallEvent/ToolResultEvent.

        Mirrors :meth:`_native_tool_loop` exactly but yields events before and
        after each tool execution.  All state writes (tool_calls,
        retrieved_chunks) are identical.
        """
        from cortex.agent.events import ToolCallEvent, ToolResultEvent
        from cortex.llm.base import SupportsToolCalling

        assert isinstance(self._llm, SupportsToolCalling)

        messages: list[AgentMessage] = [
            AgentMessage(role="user", text=state.question)
        ]

        for iteration in range(self._max_tool_calls):
            logger.debug(
                "Agent native stream loop iteration=%d user_id=%s",
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
                    "LLM generate_with_tools failed during agent stream loop",
                    details={"reason": str(exc)},
                ) from exc

            if result.is_text:
                logger.info(
                    "Agent native stream loop: text after %d tool call(s)",
                    state.total_tool_calls,
                )
                break

            tc: ToolCallRequest = result.tool_call  # type: ignore[assignment]
            tool_name = tc.tool_name

            # Emit tool_call event (before execution)
            yield ToolCallEvent(tool_name=tool_name, args=tc.args)

            messages.append(AgentMessage(role="model", tool_call=tc))

            try:
                observation = await self._registry.dispatch(
                    name=tool_name,
                    args=tc.args,
                    user=state.user,
                )
            except BadRequestError as exc:
                observation = f"Tool error: {exc.message}"
                logger.warning(
                    "Agent native stream loop: tool %r raised BadRequestError: %s",
                    tool_name,
                    exc.message,
                )

            # Accumulate RAG results
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
            await self._save_state(state, status="in_progress")

            # Emit tool_result event (bounded preview)
            yield ToolResultEvent.from_observation(tool_name, observation)

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
                "Agent native stream loop: reached max_tool_calls=%d",
                self._max_tool_calls,
            )

    async def _prompt_tool_loop_stream(
        self, *, state: AgentState
    ) -> AsyncGenerator[AgentEvent, None]:
        """Prompt-based tool-calling loop that yields ToolCallEvent/ToolResultEvent.

        Mirrors :meth:`_prompt_tool_loop` exactly but yields events.
        All state writes are identical.
        """
        from cortex.agent.events import ToolCallEvent, ToolResultEvent

        current_prompt = self._agent_prompt_builder.build_initial(
            question=state.question,
            tool_schemas=self._registry.tool_schemas,
        )

        loop_count = 0

        while loop_count < self._max_tool_calls:
            loop_count += 1
            logger.debug(
                "Agent prompt stream loop iteration=%d user_id=%s",
                loop_count,
                state.user.id,
            )

            try:
                raw_response = await self._llm.generate(current_prompt)
            except ServiceUnavailableError:
                raise
            except Exception as exc:
                raise ServiceUnavailableError(
                    "LLM generation failed during agent stream loop",
                    details={"reason": str(exc)},
                ) from exc

            decision = self._parse_decision(raw_response)
            action = decision.get("action", "")

            if action == "final_answer":
                logger.info(
                    "Agent prompt stream loop: final_answer after %d tool call(s)",
                    state.total_tool_calls,
                )
                break

            if action == "tool_call":
                tool_name = str(decision.get("tool", "")).strip()
                tool_args: dict = decision.get("args", {})

                # Emit tool_call event
                yield ToolCallEvent(tool_name=tool_name or "unknown", args=tool_args)

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
                await self._save_state(state, status="in_progress")

                # Emit tool_result event
                yield ToolResultEvent.from_observation(tool_name, observation)

                current_prompt = self._agent_prompt_builder.build_observation_turn(
                    previous_prompt=current_prompt,
                    llm_decision=raw_response,
                    tool_name=tool_name,
                    observation=observation,
                )
            else:
                logger.warning(
                    "Agent prompt stream loop: unexpected action=%r",
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

    # ------------------------------------------------------------------
    # Ephemeral state-store helpers (Phase 12)
    # ------------------------------------------------------------------

    @staticmethod
    def _state_key(
        user_id: str,
        conversation_id: str | None,
        run_id: str,
    ) -> str:
        """Construct namespaced Redis key for an agent run.

        Format: cortex:agent:run:{user_id}:{conversation_id}:{run_id}
        Stateless runs use '_stateless' for conversation_id.
        """
        conv = conversation_id or "_stateless"
        return f"cortex:agent:run:{user_id}:{conv}:{run_id}"

    @staticmethod
    def _snapshot(
        state: AgentState,
        *,
        status: str = "in_progress",
        error: str | None = None,
    ) -> dict[str, Any]:
        """Serialise AgentState to an ephemeral, JSON-safe dictionary.

        Excludes non-serialisable objects (User, raw Message rows) and large
        raw text chunks (stores chunk IDs only).
        """
        return {
            "run_id": state.run_id,
            "user_id": str(state.user.id),
            "conversation_id": state.conversation_id,
            "question": state.question,
            "started_at_iso": getattr(state, "started_at_iso", ""),
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "args": tc.args,
                    "observation": tc.observation,
                }
                for tc in state.tool_calls
            ],
            "retrieved_chunk_ids": [c.chunk_id for c in state.retrieved_chunks],
            "status": status,
            "error": error,
        }

    async def _save_state(
        self,
        state: AgentState,
        *,
        status: str = "in_progress",
        error: str | None = None,
    ) -> None:
        """Persist state snapshot to StateStore if configured; never raises."""
        if self._state_store is None:
            return
        try:
            key = self._state_key(
                user_id=str(state.user.id),
                conversation_id=state.conversation_id,
                run_id=state.run_id,
            )
            snapshot = self._snapshot(state, status=status, error=error)
            await self._state_store.save(key, snapshot, ttl_seconds=self._state_ttl)
        except Exception:
            logger.warning(
                "AgentService: failed to save state snapshot (run_id=%s)",
                state.run_id,
                exc_info=True,
            )

    async def _delete_state(self, state: AgentState) -> None:
        """Delete ephemeral state on run completion; never raises."""
        if self._state_store is None:
            return
        try:
            key = self._state_key(
                user_id=str(state.user.id),
                conversation_id=state.conversation_id,
                run_id=state.run_id,
            )
            await self._state_store.delete(key)
        except Exception:
            logger.warning(
                "AgentService: failed to delete state snapshot (run_id=%s)",
                state.run_id,
                exc_info=True,
            )

    async def load_state(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        run_id: str,
    ) -> dict[str, Any] | None:
        """Load an ephemeral agent state snapshot from the state store, if available.

        Returns None if no state store is configured or the key is not found / expired.
        """
        if self._state_store is None:
            return None
        try:
            key = self._state_key(
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
            )
            return await self._state_store.load(key)
        except Exception:
            logger.warning(
                "AgentService: failed to load state snapshot (run_id=%s)",
                run_id,
                exc_info=True,
            )
            return None
