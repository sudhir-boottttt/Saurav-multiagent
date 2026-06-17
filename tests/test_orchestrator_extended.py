"""Tests for extended AgentOrchestrator (tool invocation, context, persistence)."""

import json

from src.agents.factory import AgentWrapper, LLMConfig
from src.agents.orchestrator import AgentOrchestrator
from src.agents.tool_registry import ToolRegistry
from src.observability.cost_tracker import CostTracker


def _make_agent(agent_id="test-1", role="developer_agent"):
    return AgentWrapper(
        agent_id=agent_id, name="test", role=role,
        specialization="test", tools=["search"],
        system_prompt="You are a test agent.",
        model="test-model",
    )


def test_init_with_new_components():
    """Orchestrator accepts new optional components."""
    agents = {"a1": _make_agent("a1")}
    llm = LLMConfig()
    registry = ToolRegistry()
    tracker = CostTracker()

    orch = AgentOrchestrator(
        agents=agents, llm_config=llm,
        tool_registry=registry, cost_tracker=tracker,
    )
    assert orch._tool_registry is registry
    assert orch._cost_tracker is tracker


def test_init_backward_compat():
    """Orchestrator works without new components (backward compat)."""
    agents = {"a1": _make_agent("a1")}
    orch = AgentOrchestrator(agents=agents, llm_config=LLMConfig())
    assert orch._tool_registry is None
    assert orch._cost_tracker is None


def test_execute_tool_call():
    """Tool execution returns result as string."""
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)

    agents = {"a1": _make_agent("a1")}
    orch = AgentOrchestrator(
        agents=agents, llm_config=LLMConfig(), tool_registry=registry,
    )
    result = orch._execute_tool_call("add", json.dumps({"a": 1, "b": 2}))
    assert "3" in result


# Property 40: Tool Call Error Handling
def test_tool_call_error_handling():
    """Tool exceptions are caught and formatted as error strings."""
    registry = ToolRegistry()
    registry.register("fail", lambda: (_ for _ in ()).throw(ValueError("boom")))

    agents = {"a1": _make_agent("a1")}
    orch = AgentOrchestrator(
        agents=agents, llm_config=LLMConfig(), tool_registry=registry,
    )
    result = orch._execute_tool_call("fail", "{}")
    assert "error" in result.lower() or "boom" in result.lower()


def test_tool_not_found():
    registry = ToolRegistry()
    agents = {"a1": _make_agent("a1")}
    orch = AgentOrchestrator(
        agents=agents, llm_config=LLMConfig(), tool_registry=registry,
    )
    result = orch._execute_tool_call("nonexistent", "{}")
    assert "not found" in result.lower()


def test_no_tool_registry():
    agents = {"a1": _make_agent("a1")}
    orch = AgentOrchestrator(agents=agents, llm_config=LLMConfig())
    result = orch._execute_tool_call("anything", "{}")
    assert "not configured" in result.lower()


def test_get_tool_schemas():
    registry = ToolRegistry()
    registry.register("search", lambda q: [])

    agents = {"a1": _make_agent("a1")}
    orch = AgentOrchestrator(
        agents=agents, llm_config=LLMConfig(), tool_registry=registry,
    )
    agent = agents["a1"]
    schemas = orch._get_tool_schemas(agent)
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "search"


def test_task_persistence(tmp_path):
    from src.infrastructure.storage import FilesystemBackend
    storage = FilesystemBackend(base_path=str(tmp_path))

    agents = {"a1": _make_agent("a1")}
    orch = AgentOrchestrator(
        agents=agents, llm_config=LLMConfig(), storage=storage,
    )
    orch._task_histories["t1"] = [{"role": "user", "content": "hello"}]
    orch.persist_task_state("t1")

    # Load in new orchestrator
    orch2 = AgentOrchestrator(
        agents=agents, llm_config=LLMConfig(), storage=storage,
    )
    assert orch2.load_task_state("t1") is True
    assert len(orch2._task_histories["t1"]) == 1


def test_build_system_prompt_no_assembler():
    agents = {"a1": _make_agent("a1")}
    orch = AgentOrchestrator(agents=agents, llm_config=LLMConfig())
    prompt = orch._build_system_prompt(agents["a1"])
    assert prompt == "You are a test agent."
