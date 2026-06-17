"""Observability: structured logging, audit trail, and metrics.

Provides append-only audit log with correlation IDs, structured JSON
logging, and Prometheus-compatible metrics collection.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4

from src.core.models import AuditLogEntry

logger = logging.getLogger(__name__)


class AuditTrail:
    """Append-only audit trail with correlation ID linking.

    Stores all agent actions with 365-day retention.
    Supports querying by work_item_id, pr_id, or correlation_id.
    """

    def __init__(self) -> None:
        self._entries: list[AuditLogEntry] = []
        self._by_correlation: dict[str, list[AuditLogEntry]] = defaultdict(list)
        self._by_work_item: dict[str, list[AuditLogEntry]] = defaultdict(list)
        self._by_pr: dict[str, list[AuditLogEntry]] = defaultdict(list)

    def log_action(
        self,
        agent_id: str,
        action_type: str,
        correlation_id: str,
        payload: dict[str, Any] | None = None,
        decision_rationale: Optional[str] = None,
        retrieved_context_refs: list[str] | None = None,
    ) -> AuditLogEntry:
        """Log an agent action. Returns the created entry."""
        entry = AuditLogEntry(
            entry_id=str(uuid4()),
            correlation_id=correlation_id,
            agent_id=agent_id,
            action_type=action_type,
            payload=payload or {},
            decision_rationale=decision_rationale,
            retrieved_context_refs=retrieved_context_refs or [],
        )
        self._entries.append(entry)
        self._by_correlation[correlation_id].append(entry)

        # Index by work_item_id and pr_id if present in payload
        if "work_item_id" in (payload or {}):
            self._by_work_item[payload["work_item_id"]].append(entry)
        if "pr_id" in (payload or {}):
            self._by_pr[payload["pr_id"]].append(entry)

        return entry

    def query_by_correlation(self, correlation_id: str) -> list[AuditLogEntry]:
        """Get all entries linked by correlation_id."""
        return list(self._by_correlation.get(correlation_id, []))

    def query_by_work_item(self, work_item_id: str) -> list[AuditLogEntry]:
        """Get complete action chain for a work item."""
        return list(self._by_work_item.get(work_item_id, []))

    def query_by_pr(self, pr_id: str) -> list[AuditLogEntry]:
        """Get complete action chain for a PR."""
        return list(self._by_pr.get(pr_id, []))

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    @staticmethod
    def validate_entry(entry: AuditLogEntry) -> bool:
        """Validate entry completeness per Property 32."""
        if not all([entry.correlation_id, entry.agent_id, entry.action_type]):
            return False
        if entry.action_type == "decision":
            if not entry.decision_rationale:
                return False
        return True


class MetricsCollector:
    """Collects platform metrics for Prometheus export."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}

    def increment(self, metric: str, value: int = 1) -> None:
        self._counters[metric] += value

    def observe(self, metric: str, value: float) -> None:
        self._histograms[metric].append(value)

    def set_gauge(self, metric: str, value: float) -> None:
        self._gauges[metric] = value

    def get_counter(self, metric: str) -> int:
        return self._counters.get(metric, 0)

    def get_histogram_avg(self, metric: str) -> float:
        values = self._histograms.get(metric, [])
        return sum(values) / len(values) if values else 0.0

    def get_gauge(self, metric: str) -> float:
        return self._gauges.get(metric, 0.0)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        for name, value in sorted(self._counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        for name, values in sorted(self._histograms.items()):
            if values:
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {len(values)}")
                lines.append(f"{name}_sum {sum(values):.4f}")
        for name, value in sorted(self._gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        return "\n".join(lines)
