"""Property-based tests for core data model validation."""

import uuid

import pytest
from hypothesis import given, settings
from pydantic import ValidationError

from src.core.enums import AgentStatus, ApprovalDecision, BugSeverity
from src.core.models import (
    AgentRecord,
    ApprovalRequest,
    AuditLogEntry,
    BugReport,
    BusMessage,
    CodeChunk,
    CodeReference,
    Contribution,
    DiscussionThread,
    ImpactReport,
    RoleDefinition,
)
from tests.generators import (
    agent_record_st,
    approval_request_st,
    audit_log_entry_st,
    bug_report_st,
    bus_message_st,
    code_chunk_st,
    contribution_st,
    discussion_thread_st,
    impact_report_st,
    role_definition_st,
)


# Feature: multi-agent-platform, Property: Data Model Validation


@given(data=role_definition_st)
@settings(max_examples=100)
def test_role_definition_valid_construction(data):
    """Any RoleDefinition built from valid generator data should construct successfully."""
    role = RoleDefinition(**data)
    assert len(role.role_name) > 0
    assert len(role.tier) > 0
    assert len(role.capabilities) >= 1


@given(data=agent_record_st)
@settings(max_examples=100)
def test_agent_record_valid_construction(data):
    """Any AgentRecord built from valid generator data should construct successfully."""
    record = AgentRecord(**data)
    # agent_id should be a valid UUID
    uuid.UUID(record.agent_id)
    assert len(record.role) > 0
    assert record.status in AgentStatus


@given(data=bus_message_st)
@settings(max_examples=100)
def test_bus_message_valid_construction(data):
    """Any BusMessage built from valid generator data should construct successfully."""
    msg = BusMessage(**data)
    assert len(msg.topic) > 0
    assert len(msg.sender_id) > 0
    assert 0 <= msg.retry_count <= 3
    uuid.UUID(msg.message_id)
    uuid.UUID(msg.correlation_id)


@given(data=bug_report_st)
@settings(max_examples=100)
def test_bug_report_valid_construction(data):
    """Any BugReport built from valid generator data should construct successfully."""
    report = BugReport(**data)
    assert len(report.file_path) > 0
    assert report.line_start >= 1
    assert report.severity in BugSeverity
    assert len(report.description) > 0


@given(data=contribution_st)
@settings(max_examples=100)
def test_contribution_valid_construction(data):
    """Any Contribution built from valid generator data should construct successfully."""
    contrib = Contribution(**data)
    assert len(contrib.agent_id) > 0
    assert len(contrib.content) > 0


@given(data=discussion_thread_st)
@settings(max_examples=100)
def test_discussion_thread_valid_construction(data):
    """Any DiscussionThread built from valid generator data should construct successfully."""
    thread = DiscussionThread(**data)
    assert len(thread.topic) > 0
    assert thread.status in ("open", "synthesized", "closed")
    assert len(thread.moderator_id) > 0


@given(data=approval_request_st)
@settings(max_examples=100)
def test_approval_request_valid_construction(data):
    """Any ApprovalRequest built from valid generator data should construct successfully."""
    req = ApprovalRequest(**data)
    assert req.decision in ApprovalDecision
    assert len(req.action_type) > 0
    assert len(req.reviewer_id) > 0


@given(data=code_chunk_st)
@settings(max_examples=100)
def test_code_chunk_valid_construction(data):
    """Any CodeChunk built from valid generator data should construct successfully."""
    chunk = CodeChunk(**data)
    assert len(chunk.repo) > 0
    assert len(chunk.content) > 0
    assert chunk.start_line >= 1
    assert len(chunk.commit_hash) >= 7


@given(data=impact_report_st)
@settings(max_examples=100)
def test_impact_report_valid_construction(data):
    """Any ImpactReport built from valid generator data should construct successfully."""
    report = ImpactReport(**data)
    assert len(report.affected_repos) >= 1
    assert len(report.affected_endpoints) >= 1
    assert report.estimated_effort in ("low", "medium", "high")
    assert report.risk_level in ("low", "medium", "high", "critical")


@given(data=audit_log_entry_st)
@settings(max_examples=100)
def test_audit_log_entry_valid_construction(data):
    """Any AuditLogEntry built from valid generator data should construct successfully."""
    entry = AuditLogEntry(**data)
    assert len(entry.correlation_id) > 0
    assert len(entry.agent_id) > 0
    assert entry.action_type in ("message_send", "tool_invocation", "code_analysis", "work_item_op", "decision")


# --- Negative tests: invalid data should be rejected ---


def test_role_definition_rejects_empty_name():
    with pytest.raises(ValidationError):
        RoleDefinition(role_name="", tier="bottom", capabilities=["code_analysis"])


def test_role_definition_rejects_empty_capabilities():
    with pytest.raises(ValidationError):
        RoleDefinition(role_name="test", tier="bottom", capabilities=[])


def test_bus_message_rejects_retry_count_above_3():
    with pytest.raises(ValidationError):
        BusMessage(topic="inbound.svc.event", sender_id="agent-1", retry_count=4)


def test_bug_report_rejects_zero_line_start():
    with pytest.raises(ValidationError):
        BugReport(
            agent_id="a1", repo="r1", file_path="f.py",
            line_start=0, line_end=10, category="null_reference",
            severity=BugSeverity.HIGH, description="desc",
        )
