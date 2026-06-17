"""Agent health monitor for detecting stalled agents and tool loops.

Tracks heartbeats during task execution and triggers recovery when
agents exceed timeout or tool call limits.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.core.enums import AgentStatus

if TYPE_CHECKING:
    from src.infrastructure.agent_registry import AgentRegistry
    from src.infrastructure.storage import StorageBackend

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatEvent:
    """A heartbeat emitted by an agent during task execution."""

    agent_id: str
    task_id: str
    current_action: str  # calling_llm, executing_tool, waiting_for_human
    elapsed_time_seconds: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)


class AgentHealthMonitor:
    """Detects stalled agents and triggers recovery."""

    def __init__(
        self,
        agent_registry: Optional[AgentRegistry] = None,
        storage: Optional[StorageBackend] = None,
        heartbeat_interval: int = 30,
        heartbeat_timeout: int = 120,
        max_tool_calls_per_turn: int = 10,
    ) -> None:
        self._registry = agent_registry
        self._storage = storage
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._max_tool_calls = max_tool_calls_per_turn
        self._last_heartbeat: dict[str, float] = {}
        self._tool_call_counts: dict[str, int] = {}
        self._stalled_agents: set[str] = set()

    def record_heartbeat(self, event: HeartbeatEvent) -> None:
        """Record a heartbeat from an agent."""
        self._last_heartbeat[event.agent_id] = event.timestamp
        if event.agent_id in self._stalled_agents:
            self._stalled_agents.discard(event.agent_id)
        if self._storage:
            self._storage.save("heartbeats", event.agent_id, {
                "agent_id": event.agent_id,
                "task_id": event.task_id,
                "current_action": event.current_action,
                "timestamp": event.timestamp,
            })

    def record_tool_call(self, agent_id: str) -> None:
        """Record a tool call for an agent."""
        self._tool_call_counts[agent_id] = self._tool_call_counts.get(agent_id, 0) + 1

    def reset_tool_calls(self, agent_id: str) -> None:
        """Reset tool call counter (called when agent produces text response)."""
        self._tool_call_counts[agent_id] = 0

    def check_stalled_agents(self) -> list[str]:
        """Returns agent_ids that have exceeded heartbeat timeout."""
        now = time.monotonic()
        stalled = []
        for agent_id, last_hb in self._last_heartbeat.items():
            if (now - last_hb) > self._heartbeat_timeout:
                if agent_id not in self._stalled_agents:
                    self._stalled_agents.add(agent_id)
                    stalled.append(agent_id)
                    logger.warning("Agent %s stalled (no heartbeat for %ds)",
                                   agent_id, int(now - last_hb))
        return stalled

    def check_tool_loop(self, agent_id: str) -> bool:
        """Returns True if agent has exceeded max tool calls per turn."""
        count = self._tool_call_counts.get(agent_id, 0)
        if count > self._max_tool_calls:
            if agent_id not in self._stalled_agents:
                self._stalled_agents.add(agent_id)
                logger.warning("Agent %s in tool loop (%d calls)", agent_id, count)
            return True
        return False

    def is_stalled(self, agent_id: str) -> bool:
        """Check if an agent is currently marked as stalled."""
        return agent_id in self._stalled_agents

    def get_status(self) -> dict:
        """Get health status summary."""
        return {
            "monitored_agents": len(self._last_heartbeat),
            "stalled_agents": list(self._stalled_agents),
            "tool_call_counts": dict(self._tool_call_counts),
        }
