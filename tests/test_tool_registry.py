"""Tests for ToolRegistry."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.tool_registry import ToolRegistry


def sample_tool(name: str, count: int = 5) -> str:
    """Search for items by name."""
    return f"{name}:{count}"


def no_doc_tool(x: int) -> int:
    return x * 2


# Property 1: Tool Registry Round-Trip
@given(name=st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnop._"))
@settings(max_examples=50)
def test_tool_round_trip(name):
    """Register then lookup returns same callable."""
    registry = ToolRegistry()
    registry.register(name, sample_tool)
    found = registry.lookup(name)
    assert found is sample_tool
    assert found("test", 3) == sample_tool("test", 3)


# Property 2: Tool Schema Generation Correctness
def test_schema_generation():
    """Schema should have correct description, required, and optional params."""
    registry = ToolRegistry()
    registry.register("search", sample_tool)
    schema = registry.get_schema("search")

    func_schema = schema["function"]
    assert func_schema["name"] == "search"
    assert func_schema["description"] == "Search for items by name."

    params = func_schema["parameters"]
    assert "name" in params["properties"]
    assert "count" in params["properties"]
    assert "name" in params["required"]
    assert "count" not in params["required"]  # has default


def test_schema_no_docstring():
    """Tool without docstring should use name as description."""
    schema = ToolRegistry.generate_schema("nodoc", no_doc_tool)
    assert schema["function"]["description"]  # non-empty


# Property 3: Unregistered Tool Lookup Error
@given(name=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnop"))
@settings(max_examples=20)
def test_unregistered_lookup_error(name):
    """Unregistered tool lookup should raise KeyError with available names."""
    registry = ToolRegistry()
    registry.register("existing_tool", sample_tool)
    if name != "existing_tool":
        with pytest.raises(KeyError, match="Available tools"):
            registry.lookup(name)


def test_get_schemas_for_agent():
    registry = ToolRegistry()
    registry.register("tool_a", sample_tool)
    registry.register("tool_b", no_doc_tool)
    schemas = registry.get_schemas_for_agent(["tool_a", "tool_b"])
    assert len(schemas) == 2


def test_list_tools():
    registry = ToolRegistry()
    registry.register("b_tool", sample_tool)
    registry.register("a_tool", no_doc_tool)
    assert registry.list_tools() == ["a_tool", "b_tool"]


def test_auto_register_builtins():
    """Auto-register should register public methods from service instances."""

    class FakeService:
        def search(self, query: str) -> list:
            """Search for items."""
            return []

        def _private(self):
            pass

    registry = ToolRegistry()
    registry.auto_register_builtins(svc=FakeService())
    assert "svc.search" in registry.list_tools()
    assert all("_private" not in t for t in registry.list_tools())
