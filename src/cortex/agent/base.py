"""Abstract base class for all agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cortex.db.models.user import User


class Tool(ABC):
    """A discrete capability the agent can invoke.

    Each concrete tool must declare:

    * ``name``              — unique identifier used in tool-call JSON.
    * ``description``       — natural-language description shown to the LLM.
    * ``parameters_schema`` — JSON Schema object describing the ``args`` dict.

    The agent layer depends only on this interface; concrete implementations
    can wrap any existing service without the agent knowing the details.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case tool identifier (used in JSON action decisions)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does and when to use it."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema object for the ``args`` field in tool-call decisions.

        Example::

            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "..."}
                },
                "required": ["query"],
            }
        """

    @abstractmethod
    async def execute(self, args: dict[str, Any], user: User) -> str:
        """Execute the tool and return a plain-text observation.

        Parameters
        ----------
        args:
            Tool arguments from the LLM's action decision; validated against
            ``parameters_schema`` before this method is called.
        user:
            Authenticated owner — passed through so every tool can enforce
            user-level ownership checks independently.

        Returns
        -------
        str
            An observation string the agent appends to its reasoning context
            for the next turn.
        """
