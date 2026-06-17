"""Property tests and unit tests for Observability.

Property tests cover Property 32 from the design document.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.observability.audit import AuditTrail, MetricsCollector
from tests.generators import uuid_st, non_empty_str


# ──────────────────────────────────────────────
# Property 32: Audit Trail Completeness
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 32: Audit Trail Completeness


@given(
    agent_id=uuid_st,
    action_type=st.sampled_from(["message_send", "tool_invocation", "code_analysis", "work_item_op", "decision"]),
    correlation_id=uuid_st,
    detail=non_empty_str,
)
@settings(max_examples=100)
def test_property_32_audit_trail_completeness(agent_id, action_type, correlation_id, detail):
    """Every agent action SHALL create an audit entry with correlation_id.
    Decision actions SHALL include decision_rationale and retrieved_context_refs.
    """
    trail = AuditTrail()

    rationale = f"Decided because {detail}" if action_type == "decision" else None
    refs = [str(i) for i in range(3)] if action_type == "decision" else []

    entry = trail.log_action(
        agent_id=agent_id,
        action_type=action_type,
        correlation_id=correlation_id,
        payload={"detail": detail},
        decision_rationale=rationale,
        retrieved_context_refs=refs,
    )

    # Entry has correlation_id
    assert entry.correlation_id == correlation_id
    assert entry.agent_id == agent_id
    assert entry.action_type == action_type

    # Queryable by correlation_id
    chain = trail.query_by_correlation(correlation_id)
    assert len(chain) >= 1
    assert any(e.entry_id == entry.entry_id for e in chain)

    # Decision entries have rationale
    if action_type == "decision":
        assert entry.decision_rationale is not None
        assert len(entry.retrieved_context_refs) > 0

    # Validate
    assert AuditTrail.validate_entry(entry)


@given(
    correlation_id=uuid_st,
    work_item_id=non_empty_str,
    pr_id=non_empty_str,
)
@settings(max_examples=50)
def test_property_32_query_by_artifact(correlation_id, work_item_id, pr_id):
    """Querying by work_item_id or pr_id SHALL return the complete action chain."""
    trail = AuditTrail()

    # Log multiple related actions
    trail.log_action("dev-1", "code_analysis", correlation_id, {"work_item_id": work_item_id})
    trail.log_action("dev-1", "tool_invocation", correlation_id, {"work_item_id": work_item_id, "pr_id": pr_id})
    trail.log_action("tl-1", "decision", correlation_id,
                     {"work_item_id": work_item_id},
                     decision_rationale="Approved fix",
                     retrieved_context_refs=["ctx-1"])

    wi_chain = trail.query_by_work_item(work_item_id)
    assert len(wi_chain) == 3

    pr_chain = trail.query_by_pr(pr_id)
    assert len(pr_chain) == 1


class TestAuditTrail:

    def test_log_and_query(self):
        trail = AuditTrail()
        cid = "corr-001"
        trail.log_action("agent-1", "message_send", cid, {"msg": "hello"})
        trail.log_action("agent-2", "tool_invocation", cid, {"tool": "search"})
        chain = trail.query_by_correlation(cid)
        assert len(chain) == 2

    def test_empty_query(self):
        trail = AuditTrail()
        assert trail.query_by_correlation("nonexistent") == []
        assert trail.query_by_work_item("nonexistent") == []
        assert trail.query_by_pr("nonexistent") == []

    def test_validate_decision_without_rationale(self):
        from src.core.models import AuditLogEntry
        entry = AuditLogEntry(
            correlation_id="c1", agent_id="a1", action_type="decision",
            decision_rationale=None,
        )
        assert not AuditTrail.validate_entry(entry)

    def test_total_entries(self):
        trail = AuditTrail()
        trail.log_action("a1", "message_send", "c1")
        trail.log_action("a2", "message_send", "c2")
        assert trail.total_entries == 2


class TestMetricsCollector:

    def test_counter(self):
        mc = MetricsCollector()
        mc.increment("bugs_detected")
        mc.increment("bugs_detected")
        assert mc.get_counter("bugs_detected") == 2

    def test_histogram(self):
        mc = MetricsCollector()
        mc.observe("response_time_ms", 100.0)
        mc.observe("response_time_ms", 200.0)
        assert mc.get_histogram_avg("response_time_ms") == 150.0

    def test_gauge(self):
        mc = MetricsCollector()
        mc.set_gauge("active_agents", 5.0)
        assert mc.get_gauge("active_agents") == 5.0

    def test_prometheus_export(self):
        mc = MetricsCollector()
        mc.increment("requests_total", 10)
        mc.set_gauge("agents_healthy", 3.0)
        output = mc.export_prometheus()
        assert "requests_total 10" in output
        assert "agents_healthy 3.0" in output
