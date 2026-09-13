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
Your goal is to answer the user's question accurately.

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

1. Always respond with valid JSON — no markdown, no prose around it.
2. Call at most one tool per turn.
3. Use the retrieved information to ground your final answer.
4. If no document context is available, say so honestly.
5. Never call a tool that is not listed above.
6. Do NOT fabricate tool names or arguments.
"""

_OBSERVATION_TEMPLATE = """\
=== Tool: {tool_name} ===
{observation}
=== End of Tool Result ==="""


class AgentPromptBuilder:
    """Build prompts for the agent's reasoning loop.

    This class is **stateless** — it receives all context on each call.
    It is intentionally separate from
    :class:`~cortex.services.prompt_builder.PromptBuilder`
    which builds RAG prompts for the *final* grounded answer step.
    """

    def build_initial(
        self,
        *,
        question: str,
        tool_schemas: list[dict[str, Any]],
    ) -> str:
        """Build the initial prompt that starts the agent reasoning loop.

        Parameters
        ----------
        question:
            The user's natural-language question.
        tool_schemas:
            List of tool schema dicts from :class:`~cortex.agent.registry.ToolRegistry`.

        Returns
        -------
        str
            Full prompt with system instructions + tool schemas + question.
        """
        schemas_str = json.dumps(tool_schemas, indent=2)
        system = _AGENT_SYSTEM_PROMPT.format(tool_schemas=schemas_str)
        prompt = f"{system}\n\n=== USER QUESTION ===\n{question.strip()}"
        logger.debug(
            "Built agent initial prompt (question_len=%d, tools=%d, prompt_len=%d)",
            len(question),
            len(tool_schemas),
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
    ) -> str:
        """Extend the conversation with an LLM decision + tool observation.

        Each iteration appends the model's last JSON decision and the tool's
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


        Returns
        -------
        str
            Extended prompt for the next reasoning turn.
        """
        obs_block = _OBSERVATION_TEMPLATE.format(
            tool_name=tool_name,
            observation=observation,
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
            "Extended agent prompt with observation (tool=%s, prompt_len=%d)",
            tool_name,
            len(extended),
        )
        return extended
