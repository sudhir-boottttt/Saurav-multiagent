"""Tool Registry with auto-generated OpenAI-compatible function schemas.

Maps tool name strings to callable Python functions and generates
JSON Schema from type hints and docstrings.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Optional, get_type_hints

logger = logging.getLogger(__name__)

# Python type to JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolRegistry:
    """Maps tool name strings to callable Python functions with auto-generated schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}

    def register(self, name: str, func: Callable) -> None:
        """Register a tool by name. Auto-generates OpenAI function schema from signature."""
        self._tools[name] = func
        self._schemas[name] = self.generate_schema(name, func)
        logger.info("Registered tool: %s", name)

    def lookup(self, name: str) -> Callable:
        """Look up a tool by name. Raises KeyError with available names if not found."""
        if name not in self._tools:
            available = sorted(self._tools.keys())
            raise KeyError(
                f"Tool '{name}' not found. Available tools: {available}"
            )
        return self._tools[name]

    def get_schema(self, name: str) -> dict:
        """Get the OpenAI-compatible function schema for a tool."""
        if name not in self._schemas:
            raise KeyError(f"Tool '{name}' not registered")
        return self._schemas[name]

    def get_schemas_for_agent(self, tool_names: list[str]) -> list[dict]:
        """Return all tool schemas for a list of tool names."""
        schemas = []
        for name in tool_names:
            if name in self._schemas:
                schemas.append(self._schemas[name])
        return schemas

    def list_tools(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def auto_register_builtins(self, **services: Any) -> None:
        """Auto-register built-in tools from service instances.

        Accepts keyword arguments mapping service names to instances.
        Registers public methods as tools with dotted names (e.g., 'qdrant.search_code').
        """
        for service_name, service in services.items():
            for method_name in dir(service):
                if method_name.startswith("_"):
                    continue
                method = getattr(service, method_name)
                if callable(method):
                    tool_name = f"{service_name}.{method_name}"
                    self.register(tool_name, method)

    @staticmethod
    def generate_schema(name: str, func: Callable) -> dict:
        """Generate OpenAI function schema from Python function signature + docstring."""
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        description = doc.split("\n")[0] if doc else name

        # Get type hints, handling failures gracefully
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            # Resolve Optional types
            json_type = _resolve_json_type(param_type)

            prop: dict[str, Any] = {"type": json_type}
            properties[param_name] = prop

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return schema


def _resolve_json_type(python_type: Any) -> str:
    """Resolve a Python type annotation to a JSON Schema type string."""
    # Handle basic types
    if python_type in _TYPE_MAP:
        return _TYPE_MAP[python_type]

    # Handle Optional[X] (Union[X, None])
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        # list[X] -> array
        if origin is list:
            return "array"
        # dict[X, Y] -> object
        if origin is dict:
            return "object"

    # Handle string type annotations
    if isinstance(python_type, str):
        return "string"

    return "string"
