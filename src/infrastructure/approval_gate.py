"""Approval Gate Service for human-in-the-loop workflow control.

Blocks critical actions (merge_pr, create_prod_work_item,
cross_domain_migration, agent_config_change) until a human reviewer
approves. Supports timeout escalation and rejection routing.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Optional
from uuid import uuid4

from src.core.enums import ApprovalDecision
from src.core.models import ApprovalRequest

if TYPE_CHECKING:
    from src.infrastructure.storage import StorageBackend

logger = logging.getLogger(__name__)

# Actions that require human approval
GATED_ACTIONS = frozenset({
    "merge_pr",
    "create_prod_work_item",
    "cross_domain_migration",
    "agent_config_change",
})

DEFAULT_TIMEOUT_HOURS = 24


class ApprovalGateError(Exception):
    """Base error for approval gate operations."""


class RequestNotFoundError(ApprovalGateError):
    """Raised when an approval request doesn't exist."""


class InvalidTransitionError(ApprovalGateError):
    """Raised when an invalid state transition is attempted."""


class ApprovalGateService:
    """Manages human-in-the-loop approval gates for critical actions.

    Enforces approval for gated action types, handles timeout escalation,
    and routes rejection reasons back to originating agents.
    """

    def __init__(
        self,
        timeout_hours: int = DEFAULT_TIMEOUT_HOURS,
        notify_callback: Optional[Callable[[ApprovalRequest], None]] = None,
        storage: Optional[StorageBackend] = None,
    ) -> None:
        """
        Args:
            timeout_hours: Hours before auto-escalation (default 24).
            notify_callback: Optional callback invoked when approval is requested or escalated.
            storage: Optional StorageBackend for persisting approval requests.
        """
        self._requests: dict[str, ApprovalRequest] = {}
        self._timeout_hours = timeout_hours
        self._notify_callback = notify_callback
        self._storage = storage
        # Track rejection routing callbacks: requester_id -> callback
        self._rejection_handlers: dict[str, Callable[[str, str], None]] = {}

    def register_rejection_handler(
        self, requester_id: str, handler: Callable[[str, str], None],
    ) -> None:
        """Register a callback for when a request from this agent is rejected.

        handler(request_id, rejection_reason) is called on rejection.
        """
        self._rejection_handlers[requester_id] = handler

    def is_gated_action(self, action_type: str) -> bool:
        """Check if an action type requires approval."""
        return action_type in GATED_ACTIONS

    def request_approval(
        self,
        action_type: str,
        context: dict[str, Any],
        requester_id: str,
        reviewer_id: str,
        secondary_reviewer_id: Optional[str] = None,
        discussion_thread_id: Optional[str] = None,
    ) -> ApprovalRequest:
        """Create an approval request that blocks the workflow.

        Raises ApprovalGateError if action_type is not a gated action.
        """
        if not self.is_gated_action(action_type):
            raise ApprovalGateError(
                f"Action '{action_type}' is not a gated action. "
                f"Gated actions: {', '.join(sorted(GATED_ACTIONS))}"
            )

        if not requester_id or not requester_id.strip():
            raise ApprovalGateError("Requester ID must be non-empty")
        if not reviewer_id or not reviewer_id.strip():
            raise ApprovalGateError("Reviewer ID must be non-empty")

        now = datetime.now(UTC)
        request = ApprovalRequest(
            request_id=str(uuid4()),
            action_type=action_type,
            requester_id=requester_id,
            context=context,
            discussion_thread_id=discussion_thread_id,
            reviewer_id=reviewer_id,
            secondary_reviewer_id=secondary_reviewer_id,
            decision=ApprovalDecision.PENDING,
            rejection_reason=None,
            created_at=now,
            decided_at=None,
            deadline=now + timedelta(hours=self._timeout_hours),
        )
        self._requests[request.request_id] = request
        if self._storage:
            self._storage.save("approvals", request.request_id, request.model_dump(mode="json"))

        if self._notify_callback:
            self._notify_callback(request)

        logger.info(
            "Approval requested: %s for %s by %s (reviewer: %s)",
            request.request_id, action_type, requester_id, reviewer_id,
        )
        return request

    def submit_response(
        self,
        request_id: str,
        decision: ApprovalDecision,
        reason: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> ApprovalRequest:
        """Submit an approval or rejection response.

        Raises RequestNotFoundError if request doesn't exist.
        Raises InvalidTransitionError if request is not in PENDING/ESCALATED state.
        """
        request = self._get_request_or_raise(request_id)

        if request.decision not in (ApprovalDecision.PENDING, ApprovalDecision.ESCALATED):
            raise InvalidTransitionError(
                f"Cannot respond to request {request_id} in state '{request.decision.value}'"
            )

        if decision not in (ApprovalDecision.APPROVED, ApprovalDecision.REJECTED):
            raise InvalidTransitionError(
                f"Response must be APPROVED or REJECTED, got '{decision.value}'"
            )

        request.decision = decision
        request.decided_at = datetime.now(UTC)

        if decision == ApprovalDecision.REJECTED:
            request.rejection_reason = reason or "No reason provided"
            # Route rejection back to originating agent
            handler = self._rejection_handlers.get(request.requester_id)
            if handler:
                handler(request_id, request.rejection_reason)
            logger.info("Request %s REJECTED: %s", request_id, request.rejection_reason)
        else:
            logger.info("Request %s APPROVED", request_id)

        return request

    def get_pending(self, reviewer_id: Optional[str] = None) -> list[ApprovalRequest]:
        """Get pending approval requests, optionally filtered by reviewer."""
        pending = []
        for req in self._requests.values():
            if req.decision in (ApprovalDecision.PENDING, ApprovalDecision.ESCALATED):
                if reviewer_id is None or req.reviewer_id == reviewer_id:
                    pending.append(req)
        return pending

    def get_request(self, request_id: str) -> ApprovalRequest:
        """Get an approval request by ID."""
        return self._get_request_or_raise(request_id)

    def escalate(self, request_id: str) -> ApprovalRequest:
        """Escalate a request to the secondary reviewer.

        Used when the primary reviewer hasn't responded within the timeout.
        Extends the deadline by another timeout period.
        """
        request = self._get_request_or_raise(request_id)

        if request.decision != ApprovalDecision.PENDING:
            raise InvalidTransitionError(
                f"Cannot escalate request {request_id} in state '{request.decision.value}'"
            )

        if not request.secondary_reviewer_id:
            raise ApprovalGateError(
                f"Request {request_id} has no secondary reviewer for escalation"
            )

        request.decision = ApprovalDecision.ESCALATED
        request.reviewer_id = request.secondary_reviewer_id
        request.deadline = datetime.now(UTC) + timedelta(hours=self._timeout_hours)

        if self._notify_callback:
            self._notify_callback(request)

        logger.info(
            "Request %s escalated to %s, new deadline: %s",
            request_id, request.reviewer_id, request.deadline,
        )
        return request

    def check_timeouts(self) -> list[ApprovalRequest]:
        """Check for timed-out requests and auto-escalate them.

        Returns list of escalated requests.
        """
        now = datetime.now(UTC)
        escalated = []
        for req in list(self._requests.values()):
            if req.decision == ApprovalDecision.PENDING and now >= req.deadline:
                if req.secondary_reviewer_id:
                    self.escalate(req.request_id)
                    escalated.append(req)
                else:
                    logger.warning(
                        "Request %s timed out with no secondary reviewer", req.request_id,
                    )
        return escalated

    def _get_request_or_raise(self, request_id: str) -> ApprovalRequest:
        request = self._requests.get(request_id)
        if not request:
            raise RequestNotFoundError(f"Approval request {request_id} not found")
        return request
