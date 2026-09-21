"""Lightweight task planner for the prompt-based agent loop (Phase 14).

The ``AgentPlanner`` runs a single pre-loop LLM call that asks the model to
produce a short JSON plan - an ordered list of tool names it intends to call
before it has enough information to answer the question.

Design constraints
------------------
* **Optional** - ``AgentService`` accepts ``planner=None`` (the default) which
  disables planning entirely and preserves all Phase 8-13 behaviour.
* **Prompt-based only** - planning is irrelevant for ``SupportsToolCalling``
  providers because the native API already handles multi-step planning
  internally.  ``AgentService`` skips the planner on the native path.
* **Non-fatal** - any LLM error or JSON parse error causes the planner to
  return an empty ``TaskPlan`` and log a warning.  The tool-calling loop
  continues normally.
* **No new dependencies** - uses the same ``LLMProvider.generate`` method
  already used by the prompt-based loop.
* **Stateless** - ``AgentPlanner`` holds no per-request state; it is safe to
  share a single instance across concurrent requests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cortex.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are a planning assistant.  Given a user question and a list of available
tools, your task is to decide which tools (if any) should be called, and in
what order, before answering the question.

Available tools:
{tool_schemas}

Context flags:
- Conversation history available: {has_history}
- Long-term memory available: {has_memory}

Respond with a single valid JSON object:
{{
  "steps": ["<tool_name>", ...],
  "rationale": "<one sentence explaining your plan>",
  "confidence": <0.0 to 1.0>
}}

Rules:
1. "steps" must only contain tool names from the list above.
2. Use an empty list if the question can be answered directly without tools.
3. Do NOT fabricate tool names.
4. Keep "rationale" concise (max 100 words).
5. "confidence" is your rough estimate of how likely this plan will succeed.

User question: {question}
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TaskPlan:
    """The result of a single planning call.

    Attributes
    ----------
    steps:
        Ordered list of tool names the planner suggests calling.  May be
        empty if the question can be answered directly.
    rationale:
        One-sentence explanation of the plan (for logging and debugging).
    confidence:
        Rough planner confidence in the range ``[0, 1]``.
    """

    steps: list[str] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Planner service
# ---------------------------------------------------------------------------


class AgentPlanner:
    """Generate a brief tool-execution plan before the agent loop starts.

    Parameters
    ----------
    llm_provider:
        The same ``LLMProvider`` used by ``AgentService``.  A single extra
        ``generate()`` call is made per agent run when planning is active.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    async def plan(
        self,
        *,
        question: str,
        tool_schemas: list[dict[str, Any]],
        has_memory: bool = False,
        has_history: bool = False,
    ) -> TaskPlan:
        """Ask the LLM to produce a brief execution plan.

        Parameters
        ----------
        question:
            The current user question.
        tool_schemas:
            Tool descriptors from :class:`~cortex.agent.registry.ToolRegistry`.
        has_memory:
            ``True`` when long-term memory hits were retrieved for this run.
        has_history:
            ``True`` when previous conversation messages are available.

        Returns
        -------
        TaskPlan
            The parsed plan, or an empty ``TaskPlan`` if planning failed.
            Failure is always non-fatal.
        """
        schemas_str = json.dumps(
            [{"name": t["name"], "description": t.get("description", "")}
             for t in tool_schemas],
            indent=2,
        )
        prompt = _PLANNER_SYSTEM_PROMPT.format(
            tool_schemas=schemas_str,
            has_history=str(has_history).lower(),
            has_memory=str(has_memory).lower(),
            question=question.strip(),
        )

        try:
            raw = await self._llm.generate(prompt)
        except Exception:
            logger.warning(
                "AgentPlanner: LLM call failed; using empty plan",
                exc_info=True,
            )
            return TaskPlan()

        plan = self._parse_plan(raw, tool_schemas=tool_schemas)
        logger.debug(
            "AgentPlanner: steps=%r rationale=%r confidence=%.2f",
            plan.steps,
            plan.rationale,
            plan.confidence,
        )
        return plan

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan(
        raw: str,
        tool_schemas: list[dict[str, Any]],
    ) -> TaskPlan:
        """Parse the LLM''s JSON plan response.

        Falls back to an empty ``TaskPlan`` on any parse error.
        Filters out any tool names not in the registry to prevent prompt
        injection from influencing the plan with fabricated tool names.
        """
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()

        valid_tools = {t["name"] for t in tool_schemas}

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")  # noqa: TRY301
            raw_steps: list[str] = [
                str(s) for s in data.get("steps", []) if isinstance(s, str)
            ]
            # Filter to only registered tool names (security: reject fabricated names)
            steps = [s for s in raw_steps if s in valid_tools]
            if len(steps) < len(raw_steps):
                logger.warning(
                    "AgentPlanner: filtered %d fabricated tool name(s) from plan",
                    len(raw_steps) - len(steps),
                )
            rationale = str(data.get("rationale", ""))[:200]
            confidence_raw = data.get("confidence", 1.0)
            try:
                confidence = float(confidence_raw)
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 1.0
            return TaskPlan(steps=steps, rationale=rationale, confidence=confidence)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "AgentPlanner: could not parse LLM plan: %s (raw=%r)",
                exc,
                raw[:300],
            )
            return TaskPlan()
