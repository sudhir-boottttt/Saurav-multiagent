"""Structured decision audit trail.

Records agent decisions with context, alternatives, and rationale.
Integrates with existing AuditTrail and persists via StorageBackend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from src.observability.audit import AuditTrail

if TYPE_CHECKING:
    from src.infrastructure.storage import StorageBackend

logger = logging.getLogger(__name__)

VALID_DECISION_TYPES = {
    "task_breakdown",
    "file_selection",
    "code_change",
    "review_verdict",
    "escalation",
}


@dataclass
class DecisionEntry:
    """A structured decision record."""

    entry_id: str
    agent_id: str
    task_id: str
    timestamp: str
    decision_type: str
    context_refs: list[str]
    alternatives_considered: list[str]
    chosen_action: str
    rationale: str


class DecisionLog:
    """Structured decision audit trail integrated with existing AuditTrail."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        audit_trail: Optional[AuditTrail] = None,
    ) -> None:
        self._storage = storage
        self._audit = audit_trail
        self._entries: list[DecisionEntry] = []

    def log_decision(
        self,
        agent_id: str,
        task_id: str,
        decision_type: str,
        context_refs: list[str],
        alternatives: list[str],
        chosen_action: str,
        rationale: str,
    ) -> DecisionEntry:
        """Record a decision and emit to audit pipeline."""
        entry = DecisionEntry(
            entry_id=str(uuid4()),
            agent_id=agent_id,
            task_id=task_id,
            timestamp=datetime.now(UTC).isoformat(),
            decision_type=decision_type,
            context_refs=context_refs,
            alternatives_considered=alternatives,
            chosen_action=chosen_action,
            rationale=rationale,
        )
        self._entries.append(entry)

        if self._storage:
            self._storage.save("decisions", entry.entry_id, {
                "entry_id": entry.entry_id,
                "agent_id": entry.agent_id,
                "task_id": entry.task_id,
                "timestamp": entry.timestamp,
                "decision_type": entry.decision_type,
                "context_refs": entry.context_refs,
                "alternatives_considered": entry.alternatives_considered,
                "chosen_action": entry.chosen_action,
                "rationale": entry.rationale,
            })

        if self._audit:
            self._audit.log_action(
                agent_id=agent_id,
                action_type="decision",
                correlation_id=task_id,
                payload={
                    "decision_type": decision_type,
                    "chosen_action": chosen_action,
                },
                decision_rationale=rationale,
                retrieved_context_refs=context_refs,
            )

        logger.info("Decision logged: %s by %s for task %s", decision_type, agent_id, task_id)
        return entry

    def query_by_task(self, task_id: str) -> list[DecisionEntry]:
        return [e for e in self._entries if e.task_id == task_id]

    def query_by_agent(self, agent_id: str) -> list[DecisionEntry]:
        return [e for e in self._entries if e.agent_id == agent_id]

    def query_by_type(self, decision_type: str) -> list[DecisionEntry]:
        return [e for e in self._entries if e.decision_type == decision_type]
