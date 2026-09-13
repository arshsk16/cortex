"""Tool registry — central dispatch table for agent tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from cortex.agent.base import Tool
from cortex.core.exceptions import BadRequestError

if TYPE_CHECKING:
    from cortex.db.models.user import User

logger = logging.getLogger(__name__)

# Mapping from JSON Schema primitive type names to Python types.
# Only the types actually used in tool schemas need to be covered.
_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ToolRegistry:
    """A registry of available tools the agent can invoke.

    Tools are registered by name.  ``dispatch`` validates that the requested
    tool exists, validates ``args`` against the tool's declared JSON Schema,
    then delegates execution to the concrete implementation.

    Schema validation covers:
    * Required fields — missing required args raise :class:`BadRequestError`.
    * Type checks — each provided arg is checked against the ``type``
      declared in ``properties``.  Extra args not in ``properties`` are
      silently allowed (lenient).
    * Numeric bounds — ``minimum`` and ``maximum`` constraints are enforced
      for ``integer`` and ``number`` types.

    Individual tools may still perform their own secondary validation for
    domain-specific constraints that cannot be expressed in the schema.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry.

        Parameters
        ----------
        tool:
            A concrete :class:`~cortex.agent.base.Tool` instance.
        """
        if tool.name in self._tools:
            logger.warning(
                "Tool %r already registered — overwriting with new instance",
                tool.name,
            )
        self._tools[tool.name] = tool
        logger.debug("Registered agent tool: %s", tool.name)

    def get_tool(self, name: str) -> Tool:
        """Return the tool with the given name or raise :class:`BadRequestError`."""
        tool = self._tools.get(name)
        if tool is None:
            raise BadRequestError(
                f"Unknown tool: {name!r}",
                details={
                    "available_tools": list(self._tools.keys()),
                    "requested_tool": name,
                },
            )
        return tool

    async def dispatch(
        self,
        *,
        name: str,
        args: dict[str, Any],
        user: User,
    ) -> str:
        """Dispatch a tool call by name and return its plain-text observation.

        Validates ``args`` against the tool's ``parameters_schema`` before
        invoking the tool.  Raises :class:`BadRequestError` for missing
        required fields, type mismatches, or out-of-range numeric values.

        Parameters
        ----------
        name:
            Tool identifier — must match a registered tool's ``name``.
        args:
            Arguments from the LLM's action decision.
        user:
            Authenticated owner, forwarded to the tool for ownership checks.

        Returns
        -------
        str
            Observation text from the tool.
        """
        tool = self.get_tool(name)
        self._validate_args(
            schema=tool.parameters_schema,
            args=args,
            tool_name=name,
        )
        logger.info(
            "Agent dispatching tool=%s args=%s user_id=%s",
            name,
            args,
            user.id,
        )
        observation = await tool.execute(
            args=args,
            user=user,
        )
        logger.debug(
            "Tool %s returned observation (%d chars)",
            name,
            len(observation),
        )
        return observation

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return all registered tool schemas for injection into the system prompt."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            }
            for tool in self._tools.values()
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_args(
        schema: dict[str, Any],
        args: dict[str, Any],
        tool_name: str,
    ) -> None:
        """Validate ``args`` against a JSON Schema ``object`` descriptor.

        This is a lightweight validator covering the subset of JSON Schema
        features used by Cortex tool schemas.  It does **not** require any
        additional library dependencies.

        Validated constraints
        ---------------------
        * ``required`` — all listed fields must be present in ``args``.
        * ``properties[field].type`` — the Python type of each provided arg
          must match the declared JSON Schema type.
        * ``properties[field].minimum`` / ``maximum`` — numeric bounds.

        Extra fields not listed in ``properties`` are silently permitted
        (lenient mode) so that LLMs adding non-harmful extra keys do not
        cause hard failures.

        Parameters
        ----------
        schema:
            The tool's ``parameters_schema`` dict (top-level ``object`` type).
        args:
            The argument dict supplied by the LLM.
        tool_name:
            Used only for error message construction.

        Raises
        ------
        BadRequestError
            On any schema violation.
        """
        if schema.get("type") != "object":
            # Schema is not a simple object — skip validation; individual
            # tools are responsible for validating their own inputs.
            return

        properties: dict[str, Any] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        # --- Required-field check ---
        missing = [f for f in required if f not in args]
        if missing:
            raise BadRequestError(
                f"Tool '{tool_name}' missing required argument(s): "
                + ", ".join(f"'{f}'" for f in missing),
                details={
                    "tool": tool_name,
                    "missing_fields": missing,
                    "required": required,
                },
            )

        # --- Type and bounds checks for provided fields ---
        for field, value in args.items():
            prop_schema = properties.get(field)
            if prop_schema is None:
                # Extra field not declared in schema — allow silently
                continue

            expected_type_name: str | None = prop_schema.get("type")
            if expected_type_name is None:
                continue

            py_type = _JSON_TYPE_MAP.get(expected_type_name)
            if py_type is None:
                # Unknown JSON type — skip (forward-compatible)
                continue

            # Special case: bool is a subclass of int in Python, so
            # isinstance(True, int) is True.  Explicitly reject booleans
            # where an integer is expected and vice-versa.
            if expected_type_name == "integer" and isinstance(value, bool):
                raise BadRequestError(
                    f"Tool '{tool_name}' argument '{field}' must be an integer,"
                    " not a boolean",
                    details={
                        "tool": tool_name,
                        "field": field,
                        "expected": expected_type_name,
                        "received": "boolean",
                    },
                )
            if (
                expected_type_name == "boolean"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                raise BadRequestError(
                    f"Tool '{tool_name}' argument '{field}' must be a boolean,"
                    " not an integer",
                    details={
                        "tool": tool_name,
                        "field": field,
                        "expected": expected_type_name,
                        "received": "integer",
                    },
                )

            if not isinstance(value, py_type):
                raise BadRequestError(
                    f"Tool '{tool_name}' argument '{field}' must be of type"
                    f" '{expected_type_name}', got '{type(value).__name__}'",
                    details={
                        "tool": tool_name,
                        "field": field,
                        "expected": expected_type_name,
                        "received": type(value).__name__,
                    },
                )

            # Numeric bounds
            if expected_type_name in ("integer", "number"):
                minimum = prop_schema.get("minimum")
                maximum = prop_schema.get("maximum")
                if minimum is not None and value < minimum:
                    raise BadRequestError(
                        f"Tool '{tool_name}' argument '{field}' must be"
                        f" >= {minimum} (got {value})",
                        details={
                            "tool": tool_name,
                            "field": field,
                            "minimum": minimum,
                            "received": value,
                        },
                    )
                if maximum is not None and value > maximum:
                    raise BadRequestError(
                        f"Tool '{tool_name}' argument '{field}' must be"
                        f" <= {maximum} (got {value})",
                        details={
                            "tool": tool_name,
                            "field": field,
                            "maximum": maximum,
                            "received": value,
                        },
                    )
