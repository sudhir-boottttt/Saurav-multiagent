"""Agent performance metrics with rolling window averages.

Tracks per-agent response times, tool call success rates,
verification pass rates, and per-task outcomes.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.infrastructure.storage import StorageBackend

logger = logging.getLogger(__name__)


class AgentPerformanceMetrics:
    """Tracks per-agent and per-task performance statistics."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        rolling_window_tasks: int = 10,
        alert_verification_threshold: float = 0.5,
    ) -> None:
        self._storage = storage
        self._window = rolling_window_tasks
        self._alert_threshold = alert_verification_threshold
        self._response_times: dict[str, deque] = defaultdict(lambda: deque(maxlen=rolling_window_tasks))
        self._tool_calls: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
        self._verifications: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0})
        self._task_outcomes: list[dict] = []

    def record_response_time(self, agent_id: str, duration_ms: float) -> None:
        self._response_times[agent_id].append(duration_ms)

    def record_tool_call(self, agent_id: str, success: bool) -> None:
        key = "success" if success else "failure"
        self._tool_calls[agent_id][key] += 1

    def record_verification(self, agent_id: str, passed: bool) -> None:
        key = "passed" if passed else "failed"
        self._verifications[agent_id][key] += 1

    def record_task_outcome(
        self, task_id: str, outcome: str,
        duration_ms: float, agent_turns: int,
        tool_calls: int, hip_count: int,
    ) -> None:
        entry = {
            "task_id": task_id, "outcome": outcome,
            "duration_ms": duration_ms, "agent_turns": agent_turns,
            "tool_calls": tool_calls, "hip_count": hip_count,
        }
        self._task_outcomes.append(entry)
        if self._storage:
            self._storage.save("task_outcomes", task_id, entry)

    def get_agent_summary(self, agent_id: str) -> dict:
        times = list(self._response_times.get(agent_id, []))
        tc = self._tool_calls.get(agent_id, {"success": 0, "failure": 0})
        vr = self._verifications.get(agent_id, {"passed": 0, "failed": 0})
        total_tc = tc["success"] + tc["failure"]
        total_vr = vr["passed"] + vr["failed"]
        return {
            "agent_id": agent_id,
            "avg_response_time_ms": sum(times) / len(times) if times else 0,
            "tool_call_success_rate": tc["success"] / total_tc if total_tc else 0,
            "verification_pass_rate": vr["passed"] / total_vr if total_vr else 0,
            "total_tool_calls": total_tc,
            "total_verifications": total_vr,
        }

    def get_platform_summary(self) -> dict:
        return {
            "agents_tracked": len(self._response_times),
            "total_task_outcomes": len(self._task_outcomes),
        }

    def check_alerts(self) -> list[dict]:
        alerts = []
        for agent_id in self._verifications:
            summary = self.get_agent_summary(agent_id)
            if summary["total_verifications"] > 0 and summary["verification_pass_rate"] < self._alert_threshold:
                alerts.append({
                    "agent_id": agent_id,
                    "alert": "low_verification_pass_rate",
                    "rate": summary["verification_pass_rate"],
                    "threshold": self._alert_threshold,
                })
        return alerts
