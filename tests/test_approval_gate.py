"""Property tests and unit tests for the Approval Gate Service.

Property tests cover Properties 26-28 from the design document.
Unit tests cover approval state machine transitions.
"""

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.core.enums import ApprovalDecision
from src.infrastructure.approval_gate import (
    GATED_ACTIONS,
    ApprovalGateError,
    ApprovalGateService,
    InvalidTransitionError,
    RequestNotFoundError,
)
from tests.generators import uuid_st, non_empty_str


# ──────────────────────────────────────────────
# Property 26: Approval Gate Enforcement
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 26: Approval Gate Enforcement


@given(
    action_type=st.sampled_from(sorted(GATED_ACTIONS)),
    requester_id=uuid_st,
    reviewer_id=uuid_st,
    context_detail=non_empty_str,
    thread_id=st.one_of(st.none(), uuid_st),
)
@settings(max_examples=100)
def test_property_26_approval_gate_enforcement(action_type, requester_id, reviewer_id, context_detail, thread_id):
    """For gated actions, the Platform SHALL trigger an Approval_Gate that
    blocks the workflow until a human reviewer responds. The notification
    SHALL contain the proposed action, supporting context, and Discussion_Thread reference.
    """
    svc = ApprovalGateService()
    ctx = {"detail": context_detail}
    req = svc.request_approval(
        action_type=action_type,
        context=ctx,
        requester_id=requester_id,
        reviewer_id=reviewer_id,
        discussion_thread_id=thread_id,
    )

    # Request is created in PENDING state (blocking)
    assert req.decision == ApprovalDecision.PENDING
    assert req.action_type == action_type
    assert req.context == ctx
    assert req.requester_id == requester_id
    assert req.reviewer_id == reviewer_id
    assert req.discussion_thread_id == thread_id

    # Should appear in pending list
    pending = svc.get_pending(reviewer_id)
    assert any(p.request_id == req.request_id for p in pending)


# ──────────────────────────────────────────────
# Property 27: Approval Rejection Routing
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 27: Approval Rejection Routing


@given(
    requester_id=uuid_st,
    reviewer_id=uuid_st,
    rejection_reason=non_empty_str,
)
@settings(max_examples=100)
def test_property_27_rejection_routing(requester_id, reviewer_id, rejection_reason):
    """On rejection, the rejection reason SHALL be routed back to the
    originating agent, and the workflow SHALL not proceed.
    """
    routed_rejections = []

    def on_rejection(req_id: str, reason: str):
        routed_rejections.append((req_id, reason))

    svc = ApprovalGateService()
    svc.register_rejection_handler(requester_id, on_rejection)

    req = svc.request_approval(
        action_type="merge_pr",
        context={"pr_url": "https://github.com/org/repo/pull/1"},
        requester_id=requester_id,
        reviewer_id=reviewer_id,
    )

    svc.submit_response(req.request_id, ApprovalDecision.REJECTED, reason=rejection_reason)

    # Rejection was routed to the originating agent
    assert len(routed_rejections) == 1
    assert routed_rejections[0][0] == req.request_id
    assert routed_rejections[0][1] == rejection_reason

    # Request is in REJECTED state — workflow blocked
    updated = svc.get_request(req.request_id)
    assert updated.decision == ApprovalDecision.REJECTED
    assert updated.rejection_reason == rejection_reason


# ──────────────────────────────────────────────
# Property 28: Approval Gate Escalation
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 28: Approval Gate Escalation


@given(
    requester_id=uuid_st,
    primary_reviewer=uuid_st,
    secondary_reviewer=uuid_st,
)
@settings(max_examples=100)
def test_property_28_escalation(requester_id, primary_reviewer, secondary_reviewer):
    """If no response within 24h, the Platform SHALL escalate to a
    secondary reviewer and extend the deadline by 24h.
    """
    svc = ApprovalGateService(timeout_hours=24)

    req = svc.request_approval(
        action_type="cross_domain_migration",
        context={"plan": "migrate shared API"},
        requester_id=requester_id,
        reviewer_id=primary_reviewer,
        secondary_reviewer_id=secondary_reviewer,
    )

    original_deadline = req.deadline

    # Escalate
    escalated = svc.escalate(req.request_id)

    assert escalated.decision == ApprovalDecision.ESCALATED
    assert escalated.reviewer_id == secondary_reviewer
    assert escalated.deadline > original_deadline

    # Can still approve after escalation
    svc.submit_response(req.request_id, ApprovalDecision.APPROVED)
    assert svc.get_request(req.request_id).decision == ApprovalDecision.APPROVED


# ──────────────────────────────────────────────
# Unit Tests: Approval State Machine
# ──────────────────────────────────────────────


class TestApprovalGateStateMachine:
    """Unit tests for approval gate state transitions and edge cases."""

    def test_non_gated_action_raises(self):
        svc = ApprovalGateService()
        with pytest.raises(ApprovalGateError, match="not a gated action"):
            svc.request_approval("read_file", {}, "agent-1", "reviewer-1")

    def test_all_gated_actions_accepted(self):
        svc = ApprovalGateService()
        for action in GATED_ACTIONS:
            req = svc.request_approval(action, {}, "agent-1", "reviewer-1")
            assert req.decision == ApprovalDecision.PENDING

    def test_approve_flow(self):
        svc = ApprovalGateService()
        req = svc.request_approval("merge_pr", {"pr": 1}, "dev-001", "reviewer-1")
        result = svc.submit_response(req.request_id, ApprovalDecision.APPROVED)
        assert result.decision == ApprovalDecision.APPROVED
        assert result.decided_at is not None

    def test_reject_flow(self):
        svc = ApprovalGateService()
        req = svc.request_approval("merge_pr", {"pr": 1}, "dev-001", "reviewer-1")
        result = svc.submit_response(req.request_id, ApprovalDecision.REJECTED, "Needs more tests")
        assert result.decision == ApprovalDecision.REJECTED
        assert result.rejection_reason == "Needs more tests"

    def test_cannot_respond_to_already_decided(self):
        svc = ApprovalGateService()
        req = svc.request_approval("merge_pr", {}, "dev-001", "reviewer-1")
        svc.submit_response(req.request_id, ApprovalDecision.APPROVED)
        with pytest.raises(InvalidTransitionError):
            svc.submit_response(req.request_id, ApprovalDecision.REJECTED)

    def test_cannot_escalate_non_pending(self):
        svc = ApprovalGateService()
        req = svc.request_approval("merge_pr", {}, "dev-001", "reviewer-1", secondary_reviewer_id="reviewer-2")
        svc.submit_response(req.request_id, ApprovalDecision.APPROVED)
        with pytest.raises(InvalidTransitionError):
            svc.escalate(req.request_id)

    def test_escalate_without_secondary_raises(self):
        svc = ApprovalGateService()
        req = svc.request_approval("merge_pr", {}, "dev-001", "reviewer-1")
        with pytest.raises(ApprovalGateError, match="no secondary reviewer"):
            svc.escalate(req.request_id)

    def test_get_pending_filters_by_reviewer(self):
        svc = ApprovalGateService()
        svc.request_approval("merge_pr", {}, "dev-001", "reviewer-A")
        svc.request_approval("merge_pr", {}, "dev-002", "reviewer-B")
        pending_a = svc.get_pending("reviewer-A")
        pending_b = svc.get_pending("reviewer-B")
        assert len(pending_a) == 1
        assert len(pending_b) == 1
        assert pending_a[0].reviewer_id == "reviewer-A"

    def test_get_pending_all(self):
        svc = ApprovalGateService()
        svc.request_approval("merge_pr", {}, "dev-001", "reviewer-A")
        svc.request_approval("merge_pr", {}, "dev-002", "reviewer-B")
        all_pending = svc.get_pending()
        assert len(all_pending) == 2

    def test_nonexistent_request_raises(self):
        svc = ApprovalGateService()
        with pytest.raises(RequestNotFoundError):
            svc.get_request("nonexistent")

    def test_notify_callback_called(self):
        notifications = []
        svc = ApprovalGateService(notify_callback=lambda r: notifications.append(r))
        svc.request_approval("merge_pr", {}, "dev-001", "reviewer-1")
        assert len(notifications) == 1

    def test_empty_requester_raises(self):
        svc = ApprovalGateService()
        with pytest.raises(ApprovalGateError, match="Requester ID"):
            svc.request_approval("merge_pr", {}, "", "reviewer-1")

    def test_empty_reviewer_raises(self):
        svc = ApprovalGateService()
        with pytest.raises(ApprovalGateError, match="Reviewer ID"):
            svc.request_approval("merge_pr", {}, "dev-001", "")

    def test_check_timeouts_escalates_expired(self):
        svc = ApprovalGateService(timeout_hours=0)  # Immediate timeout
        req = svc.request_approval(
            "merge_pr", {}, "dev-001", "reviewer-1", secondary_reviewer_id="reviewer-2",
        )
        # Force deadline to past
        req.deadline = datetime.now(UTC) - timedelta(hours=1)
        escalated = svc.check_timeouts()
        assert len(escalated) == 1
        assert escalated[0].decision == ApprovalDecision.ESCALATED

    def test_rejection_default_reason(self):
        svc = ApprovalGateService()
        req = svc.request_approval("merge_pr", {}, "dev-001", "reviewer-1")
        svc.submit_response(req.request_id, ApprovalDecision.REJECTED)
        assert svc.get_request(req.request_id).rejection_reason == "No reason provided"
