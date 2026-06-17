"""Scoping confirmation gate for human approval of task scope.

Pauses workflow after PM+TL scoping for human confirmation.
Supports approve/modify/reject responses.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.infrastructure.human_interaction import HumanInteractionPoint

logger = logging.getLogger(__name__)


class ScopingConfirmationGate:
    """Pauses workflow after scoping for human confirmation."""

    def __init__(
        self,
        hip: Optional[HumanInteractionPoint] = None,
        enabled: bool = True,
    ) -> None:
        self._hip = hip
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def present_scope(self, scope: dict, task_id: str, agent_id: str = "system") -> Optional[str]:
        """Format and present scope to human. Returns HIP id for tracking."""
        if not self._enabled:
            return None

        if not self._hip:
            logger.warning("ScopingConfirmationGate: no HIP configured, auto-approving")
            return None

        summary = self.format_scope_summary(scope)
        hip_record = self._hip.request_input(
            question=f"Please review the following scope and respond with 'approve', 'modify', or 'reject':\n\n{summary}",
            context=scope,
            urgency="normal",
            agent_id=agent_id,
            task_id=task_id,
        )
        return hip_record.hip_id

    def check_decision(self, hip_id: str) -> Optional[str]:
        """Check if human has responded. Returns 'approve'/'modify'/'reject' or None."""
        if not self._hip:
            return "approve"

        response = self._hip.check_response(hip_id)
        if response is None:
            return None

        response_lower = response.strip().lower()
        if response_lower in ("approve", "approved", "yes", "ok", "lgtm"):
            return "approve"
        elif response_lower in ("reject", "rejected", "no"):
            return "reject"
        else:
            return "modify"

    @staticmethod
    def format_scope_summary(scope: dict) -> str:
        """Format scope into human-readable summary."""
        parts = []
        if "task" in scope:
            parts.append(f"Task: {scope['task']}")
        if "domains" in scope:
            parts.append(f"Domains: {', '.join(scope['domains'])}")
        if "repos" in scope:
            parts.append(f"Repositories: {', '.join(scope['repos'])}")
        if "subtasks" in scope:
            parts.append("Sub-tasks:")
            for st in scope["subtasks"]:
                parts.append(f"  - {st}")
        if "estimated_effort" in scope:
            parts.append(f"Estimated effort: {scope['estimated_effort']}")
        return "\n".join(parts) if parts else str(scope)
