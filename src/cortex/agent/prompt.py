"""Agent prompt construction."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent system prompt template
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = """\
You are an intelligent assistant with access to a set of tools.
Your goal is to answer the user''s question accurately.

## Available Tools

{tool_schemas}

## Response Format

You MUST respond with a single valid JSON object on every turn.

If you need to call a tool:
{{
  "action": "tool_call",
  "tool": "<tool_name>",
  "args": {{ ... }}
}}

If you have enough information to answer the final question:
{{
  "action": "final_answer",
  "answer": "<your complete answer here>"
}}

## Rules

1. Always respond with valid JSON - no markdown, no prose around it.
2. Call at most one tool per turn.
3. Use the retrieved information to ground your final answer.
4. If no document context is available, say so honestly.
5. Never call a tool that is not listed above.
6. Do NOT fabricate tool names or arguments.
"""

_CONTEXT_FLAGS_BLOCK = """\

## Available Context

The following context is already available to assist you:
{flags_lines}

Use tools to gather additional evidence only when the above context is
insufficient to answer the question confidently.
"""

_OBSERVATION_TEMPLATE = """\
=== Tool: {tool_name} ===
{observation}
=== End of Tool Result ==="""

_OBSERVATION_TRUNCATED_SUFFIX = "\n[... observation truncated ...]"

_BUDGET_EXHAUSTED_PROMPT = """\
{previous_prompt}

=== TOOL BUDGET EXHAUSTED ===
You have used all {max_calls} allowed tool call(s).  You must now produce a
final answer using only the tool observations already available above.  If
the information is incomplete, acknowledge the limitation honestly rather
than fabricating missing details.

Respond with:
{{
  "action": "final_answer",
  "answer": "<your best answer given the available information>"
}}
"""

# Default maximum characters for a single tool observation injected into the
# prompt.  Large RAG chunks are truncated here (not just in the SSE layer) to
# prevent unbounded context growth.
_DEFAULT_MAX_OBSERVATION_CHARS = 4000


class AgentPromptBuilder:
    """Build prompts for the agent''s reasoning loop.

    This class is **stateless** - it receives all context on each call.
    It is intentionally separate from
    :class:`~cortex.services.prompt_builder.PromptBuilder`
    which builds RAG prompts for the *final* grounded answer step.

    Phase 14 additions
    ------------------
    * ``build_initial`` accepts optional ``context_flags`` to append a
      ``## Available Context`` block to the system prompt.
    * ``build_observation_turn`` accepts ``max_observation_chars`` to cap
      observation length before prompt injection.
    * ``build_budget_exhausted_turn`` signals to the LLM that the tool
      budget is exhausted and it must answer with what it has.
    """

    def build_initial(
        self,
        *,
        question: str,
        tool_schemas: list[dict[str, Any]],
        context_flags: dict[str, bool] | None = None,
    ) -> str:
        """Build the initial prompt that starts the agent reasoning loop.

        Parameters
        ----------
        question:
            The user''s natural-language question.
        tool_schemas:
            List of tool schema dicts from :class:`~cortex.agent.registry.ToolRegistry`.
        context_flags:
            Optional mapping of context availability flags, e.g.::

                {"has_memory": True, "has_history": False}

            When provided, a ``## Available Context`` block is appended to
            the system prompt so the LLM can make better tool-selection
            decisions.

        Returns
        -------
        str
            Full prompt with system instructions + tool schemas + optional
            context block + question.
        """
        schemas_str = json.dumps(tool_schemas, indent=2)
        system = _AGENT_SYSTEM_PROMPT.format(tool_schemas=schemas_str)

        if context_flags:
            flag_lines = []
            if context_flags.get("has_history"):
                flag_lines.append(
                    "- Conversation history: YES (previous exchanges available)"
                )
            else:
                flag_lines.append(
                    "- Conversation history: NO (first message or stateless run)"
                )
            if context_flags.get("has_memory"):
                flag_lines.append(
                    "- Long-term memory: YES (relevant memories already retrieved)"
                )
            else:
                flag_lines.append("- Long-term memory: NO")
            system += _CONTEXT_FLAGS_BLOCK.format(flags_lines="\n".join(flag_lines))

        prompt = f"{system}\n\n=== USER QUESTION ===\n{question.strip()}"
        logger.debug(
            "Built agent initial prompt (question_len=%d, tools=%d, "
            "context_flags=%s, prompt_len=%d)",
            len(question),
            len(tool_schemas),
            context_flags,
            len(prompt),
        )
        return prompt

    def build_observation_turn(
        self,
        *,
        previous_prompt: str,
        llm_decision: str,
        tool_name: str,
        observation: str,
        max_observation_chars: int = _DEFAULT_MAX_OBSERVATION_CHARS,
    ) -> str:
        """Extend the conversation with an LLM decision + tool observation.

        Each iteration appends the model''s last JSON decision and the tool''s
        observation before asking the model to decide again.

        Parameters
        ----------
        previous_prompt:
            The prompt from the previous turn (grows per iteration).
        llm_decision:
            The raw JSON string the LLM returned on the last turn.
        tool_name:
            Name of the tool that was just called.
        observation:
            Plain-text result from the tool.
        max_observation_chars:
            Maximum characters of ``observation`` injected into the prompt.
            Prevents runaway context growth from large RAG chunks.
            Defaults to :data:`_DEFAULT_MAX_OBSERVATION_CHARS` (4000).

        Returns
        -------
        str
            Extended prompt for the next reasoning turn.
        """
        # Truncate observation at prompt level (SSE layer has its own 500-char cap)
        if len(observation) > max_observation_chars:
            obs_injected = (
                observation[:max_observation_chars] + _OBSERVATION_TRUNCATED_SUFFIX
            )
        else:
            obs_injected = observation

        obs_block = _OBSERVATION_TEMPLATE.format(
            tool_name=tool_name,
            observation=obs_injected,
        )
        extended = (
            f"{previous_prompt}\n\n"
            f"=== ASSISTANT (previous turn) ===\n{llm_decision}\n\n"
            f"{obs_block}\n\n"
            f"=== CONTINUE ===\n"
            f"Based on the tool result above, what is your next action? "
            f"Respond with a JSON object."
        )
        logger.debug(
            "Extended agent prompt with observation "
            "(tool=%s, obs_chars=%d, prompt_len=%d)",
            tool_name,
            len(obs_injected),
            len(extended),
        )
        return extended

    def build_budget_exhausted_turn(
        self,
        *,
        previous_prompt: str,
        max_calls: int,
    ) -> str:
        """Build a prompt signalling that the tool budget is exhausted.

        Called when the tool-calling loop terminates due to hitting
        ``max_tool_calls``.  Instructs the LLM to synthesise a best-effort
        answer from the observations already gathered.

        Parameters
        ----------
        previous_prompt:
            The accumulated prompt after all tool calls.
        max_calls:
            The maximum number of tool calls that was configured.

        Returns
        -------
        str
            A prompt that forces the LLM to produce a ``final_answer``.
        """
        prompt = _BUDGET_EXHAUSTED_PROMPT.format(
            previous_prompt=previous_prompt,
            max_calls=max_calls,
        )
        logger.debug(
            "Built budget-exhausted prompt (max_calls=%d, prompt_len=%d)",
            max_calls,
            len(prompt),
        )
        return prompt
