"""Tests for AgentHealthMonitor."""

import time

from src.infrastructure.health_monitor import AgentHealthMonitor, HeartbeatEvent


def test_record_heartbeat():
    mon = AgentHealthMonitor()
    event = HeartbeatEvent(agent_id="a1", task_id="t1", current_action="calling_llm")
    mon.record_heartbeat(event)
    assert not mon.is_stalled("a1")


# Property 29: Agent Health Stall Detection
def test_stall_detection_timeout():
    """Agent without heartbeat for timeout should be marked stalled."""
    mon = AgentHealthMonitor(heartbeat_timeout=0)  # immediate timeout
    event = HeartbeatEvent(
        agent_id="a1", task_id="t1", current_action="calling_llm",
        timestamp=time.monotonic() - 10,  # 10 seconds ago
    )
    mon.record_heartbeat(event)
    stalled = mon.check_stalled_agents()
    assert "a1" in stalled
    assert mon.is_stalled("a1")


def test_stall_detection_tool_loop():
    """Agent exceeding max tool calls should be detected."""
    mon = AgentHealthMonitor(max_tool_calls_per_turn=3)
    for _ in range(4):
        mon.record_tool_call("a1")
    assert mon.check_tool_loop("a1") is True


def test_no_stall_within_timeout():
    mon = AgentHealthMonitor(heartbeat_timeout=300)
    event = HeartbeatEvent(agent_id="a1", task_id="t1", current_action="calling_llm")
    mon.record_heartbeat(event)
    stalled = mon.check_stalled_agents()
    assert "a1" not in stalled


def test_reset_tool_calls():
    mon = AgentHealthMonitor(max_tool_calls_per_turn=3)
    for _ in range(2):
        mon.record_tool_call("a1")
    mon.reset_tool_calls("a1")
    assert mon.check_tool_loop("a1") is False


def test_heartbeat_clears_stalled():
    mon = AgentHealthMonitor(heartbeat_timeout=0)
    old_event = HeartbeatEvent(
        agent_id="a1", task_id="t1", current_action="calling_llm",
        timestamp=time.monotonic() - 10,
    )
    mon.record_heartbeat(old_event)
    mon.check_stalled_agents()
    assert mon.is_stalled("a1")

    # New heartbeat should clear stalled status
    new_event = HeartbeatEvent(agent_id="a1", task_id="t1", current_action="calling_llm")
    mon.record_heartbeat(new_event)
    assert not mon.is_stalled("a1")


def test_get_status():
    mon = AgentHealthMonitor()
    event = HeartbeatEvent(agent_id="a1", task_id="t1", current_action="calling_llm")
    mon.record_heartbeat(event)
    status = mon.get_status()
    assert status["monitored_agents"] == 1
    assert status["stalled_agents"] == []
