"""Cost tracker for LLM token usage and budget enforcement.

Logs per-call token usage, aggregates by agent/task/time window,
and enforces configurable budget limits.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from src.infrastructure.storage import StorageBackend

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded."""

    def __init__(self, limit_type: str, current: int, limit: int):
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        super().__init__(f"{limit_type} budget exceeded: {current}/{limit} tokens")


class CostTracker:
    """Tracks LLM token usage and enforces budget limits."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        max_tokens_per_task: int = 500_000,
        max_tokens_per_agent_per_hour: int = 200_000,
        max_tokens_per_day: int = 5_000_000,
    ) -> None:
        self._storage = storage
        self._max_per_task = max_tokens_per_task
        self._max_per_agent_hour = max_tokens_per_agent_per_hour
        self._max_per_day = max_tokens_per_day
        self._records: list[dict[str, Any]] = []
        self._task_totals: dict[str, int] = defaultdict(int)
        self._agent_totals: dict[str, int] = defaultdict(int)
        self._daily_total: int = 0
        self._current_day: str = datetime.now(UTC).strftime("%Y-%m-%d")

    def log_usage(
        self,
        agent_id: str,
        task_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Log token usage for an LLM call."""
        total = input_tokens + output_tokens
        now = datetime.now(UTC)
        today = now.strftime("%Y-%m-%d")

        # Reset daily counter on new day
        if today != self._current_day:
            self._daily_total = 0
            self._current_day = today

        record = {
            "record_id": str(uuid4()),
            "agent_id": agent_id,
            "task_id": task_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "timestamp": now.isoformat(),
        }
        self._records.append(record)
        self._task_totals[task_id] += total
        self._agent_totals[agent_id] += total
        self._daily_total += total

        if self._storage:
            self._storage.save("cost_records", record["record_id"], record)

    def check_task_budget(self, task_id: str) -> bool:
        """Returns True if task is within budget."""
        return self._task_totals.get(task_id, 0) < self._max_per_task

    def check_agent_hourly_limit(self, agent_id: str) -> bool:
        """Returns True if agent is within hourly limit."""
        cutoff = datetime.now(UTC) - timedelta(hours=1)
        hourly = sum(
            r["total_tokens"] for r in self._records
            if r["agent_id"] == agent_id
            and datetime.fromisoformat(r["timestamp"]) >= cutoff
        )
        return hourly < self._max_per_agent_hour

    def check_daily_limit(self) -> bool:
        """Returns True if daily limit not exceeded."""
        return self._daily_total < self._max_per_day

    def get_task_usage(self, task_id: str) -> int:
        """Get total tokens used by a task."""
        return self._task_totals.get(task_id, 0)

    def get_agent_usage(self, agent_id: str) -> int:
        """Get total tokens used by an agent."""
        return self._agent_totals.get(agent_id, 0)

    def get_usage_summary(self) -> dict:
        """Get current day's usage breakdown."""
        by_agent: dict[str, int] = defaultdict(int)
        by_task: dict[str, int] = defaultdict(int)
        for r in self._records:
            by_agent[r["agent_id"]] += r["total_tokens"]
            if r["task_id"]:
                by_task[r["task_id"]] += r["total_tokens"]
        return {
            "daily_total": self._daily_total,
            "daily_limit": self._max_per_day,
            "by_agent": dict(by_agent),
            "by_task": dict(by_task),
            "total_calls": len(self._records),
        }
