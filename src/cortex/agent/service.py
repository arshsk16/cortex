"""Agent orchestration service — single-agent + tool-calling loop.

Architecture (Phase 9)
----------------------
``AgentService.run()`` has three shared sections that are always executed:

1. **Security gate** — conversation ownership check + user-message persist.
2. **Tool-calling loop** — dispatched to one of two strategies:

   * **Native path** (preferred) — used when the ``LLMProvider`` implements
     :class:`~cortex.llm.base.SupportsToolCalling`.  Maintains a typed
     message history and calls
     :meth:`~cortex.llm.base.SupportsToolCalling.generate_with_tools`
     so the provider can use its own structured function-calling API
     (e.g. Gemini ``FunctionDeclaration``).

   * **Prompt-based fallback** — used when the provider does *not* implement
     ``SupportsToolCalling``.  Embeds tool schemas in the system prompt,
     parses the LLM's JSON decision string, and feeds observations back as
     context.  This is identical to the Phase 8 loop.

3. **Grounded final answer** — :class:`~cortex.services.prompt_builder.PromptBuilder`
   constructs a grounded prompt from the accumulated retrieval chunks, then
   :class:`~cortex.llm.base.LLMProvider` generates the final citation-grounded
   answer — **identical path to RAGService**.

Provider agnosticism
--------------------
``AgentService`` imports only from :mod:`cortex.agent.types`, never from
the Gemini SDK.  Gemini-specific types live entirely inside
:class:`~cortex.llm.gemini.GeminiProvider`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING

from cortex.agent.prompt import AgentPromptBuilder
from cortex.agent.registry import ToolRegistry
from cortex.agent.result import AgentResult, ToolCallRecord
from cortex.agent.tools.rag_search import RAGSearchTool
from cortex.agent.types import AgentMessage, ToolCallRequest, ToolResult
from cortex.core.exceptions import BadRequestError, ServiceUnavailableError
from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from cortex.db.models.user import User
    from cortex.llm.base import LLMProvider
    from cortex.services.conversation import ConversationService

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOOL_CALLS = 5


class AgentService:
    """Orchestrate a single-agent + tool-calling loop.

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
        Optional — when supplied, the agent enforces conversation ownership
        and persists the user/assistant message pair.
    max_tool_calls:
        Maximum number of tool invocations per agent run.  Prevents infinite
        loops in case the LLM keeps requesting tools.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        conversation_service: ConversationService | None = None,
        max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
    ) -> None:
        self._llm = llm_provider
        self._registry = tool_registry
        self._prompt_builder = prompt_builder
        self._conv_service = conversation_service
        self._max_tool_calls = max_tool_calls
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
            When set, ownership is verified against this conversation before
            any processing, and the user/assistant messages are persisted on
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

        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 0: Conversation ownership check (security gate)  [SHARED]
        # ------------------------------------------------------------------
        if conversation_id and self._conv_service:
            await self._conv_service.get_conversation(
                conversation_id=conversation_id,
                user_id=user.id,
            )

        # ------------------------------------------------------------------
        # Step 1: Persist user message  [SHARED]
        # ------------------------------------------------------------------
        if conversation_id and self._conv_service:
            await self._conv_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            await self._conv_service.set_auto_title_if_needed(
                conversation_id=conversation_id,
                first_message=question,
            )

        # ------------------------------------------------------------------
        # Step 2: Tool-calling loop — dispatched by provider capability
        # ------------------------------------------------------------------
        from cortex.llm.base import SupportsToolCalling  # local import avoids cycles

        if isinstance(self._llm, SupportsToolCalling):
            logger.debug(
                "Agent using native tool-calling path (provider=%s)",
                type(self._llm).__name__,
            )
            tool_calls, all_retrieved_chunks = await self._native_tool_loop(
                question=question, user=user
            )
        else:
            logger.debug(
                "Agent using prompt-based tool-calling path (provider=%s)",
                type(self._llm).__name__,
            )
            tool_calls, all_retrieved_chunks = await self._prompt_tool_loop(
                question=question, user=user
            )

        # ------------------------------------------------------------------
        # Step 3: Generate grounded final answer via existing RAG path  [SHARED]
        # ------------------------------------------------------------------
        grounded_prompt = self._prompt_builder.build(
            question=question,
            retrieved_chunks=all_retrieved_chunks,
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

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Agent run complete: user_id=%s conversation_id=%s "
            "tool_calls=%d chunks=%d total_ms=%.1f answer_len=%d",
            user.id,
            conversation_id,
            len(tool_calls),
            len(all_retrieved_chunks),
            total_ms,
            len(final_answer),
        )

        # ------------------------------------------------------------------
        # Step 4: Persist assistant message + token usage (optional)  [SHARED]
        # ------------------------------------------------------------------
        if conversation_id and self._conv_service:
            citation_dicts = [
                {
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                }
                for c in all_retrieved_chunks
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
            tool_calls=tool_calls,
            retrieved_chunks=all_retrieved_chunks,
        )

    # ------------------------------------------------------------------
    # Private — Native tool-calling loop (Phase 9)
    # ------------------------------------------------------------------

    async def _native_tool_loop(
        self,
        *,
        question: str,
        user: User,
    ) -> tuple[list[ToolCallRecord], list[RetrievalResult]]:
        """Run the tool-calling loop using the provider's native API.

        Maintains a typed :class:`~cortex.agent.types.AgentMessage` history
        and calls :meth:`~cortex.llm.base.SupportsToolCalling.generate_with_tools`
        on each iteration.  All Gemini-specific types remain inside the
        provider; this method is fully provider-agnostic.

        Returns
        -------
        tuple[list[ToolCallRecord], list[RetrievalResult]]
            Tool call trace and accumulated retrieval results.
        """
        from cortex.llm.base import SupportsToolCalling

        assert isinstance(self._llm, SupportsToolCalling)  # guaranteed by caller

        tool_calls: list[ToolCallRecord] = []
        all_retrieved_chunks: list[RetrievalResult] = []

        # Seed conversation history with the user's question
        messages: list[AgentMessage] = [AgentMessage(role="user", text=question)]

        for iteration in range(self._max_tool_calls):
            logger.debug(
                "Agent native loop iteration=%d user_id=%s",
                iteration + 1,
                user.id,
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
                # LLM produced a text response — tool loop is done
                logger.info(
                    "Agent native loop: text response after %d tool call(s)",
                    len(tool_calls),
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
                    user=user,
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
                all_retrieved_chunks.extend(tool_obj.last_results)

            tool_calls.append(
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

        return tool_calls, all_retrieved_chunks

    # ------------------------------------------------------------------
    # Private — Prompt-based tool-calling loop (Phase 8 fallback)
    # ------------------------------------------------------------------

    async def _prompt_tool_loop(
        self,
        *,
        question: str,
        user: User,
    ) -> tuple[list[ToolCallRecord], list[RetrievalResult]]:
        """Prompt-based tool-calling loop (Phase 8 fallback).

        Used when the LLM provider does **not** implement
        :class:`~cortex.llm.base.SupportsToolCalling`.  Embeds tool schemas
        in the system prompt and parses the LLM's JSON decisions.

        Returns
        -------
        tuple[list[ToolCallRecord], list[RetrievalResult]]
            Tool call trace and accumulated retrieval results.
        """
        tool_calls: list[ToolCallRecord] = []
        all_retrieved_chunks: list[RetrievalResult] = []

        current_prompt = self._agent_prompt_builder.build_initial(
            question=question,
            tool_schemas=self._registry.tool_schemas,
        )

        loop_count = 0

        while loop_count < self._max_tool_calls:
            loop_count += 1
            logger.debug(
                "Agent prompt loop iteration=%d user_id=%s",
                loop_count,
                user.id,
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
                    len(tool_calls),
                )
                break

            if action == "tool_call":
                tool_name = str(decision.get("tool", "")).strip()
                tool_args: dict = decision.get("args", {})

                try:
                    observation = await self._registry.dispatch(
                        name=tool_name,
                        args=tool_args,
                        user=user,
                    )
                except BadRequestError as exc:
                    observation = f"Tool error: {exc.message}"
                    tool_name = tool_name or "unknown"

                tool_obj = None
                with contextlib.suppress(BadRequestError):
                    tool_obj = self._registry.get_tool(tool_name)
                if isinstance(tool_obj, RAGSearchTool):
                    all_retrieved_chunks.extend(tool_obj.last_results)

                tool_calls.append(
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

        return tool_calls, all_retrieved_chunks

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
            inner_lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
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
