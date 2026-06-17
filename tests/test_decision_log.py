"""Tests for DecisionLog."""

from src.observability.audit import AuditTrail
from src.observability.decision_log import DecisionLog


def test_log_decision():
    dl = DecisionLog()
    entry = dl.log_decision(
        agent_id="dev-1",
        task_id="task-1",
        decision_type="file_selection",
        context_refs=["src/main.py"],
        alternatives=["src/utils.py", "src/handler.py"],
        chosen_action="Edit src/main.py",
        rationale="Main entry point for the feature",
    )
    assert entry.agent_id == "dev-1"
    assert entry.decision_type == "file_selection"
    assert entry.chosen_action == "Edit src/main.py"
    assert entry.entry_id  # auto-generated


# Property 36: Decision Log Completeness
def test_decision_log_completeness():
    """Every logged decision has required fields."""
    dl = DecisionLog()
    entry = dl.log_decision("a1", "t1", "code_change", ["f.py"], ["alt1"], "action", "reason")
    assert entry.agent_id == "a1"
    assert entry.task_id == "t1"
    assert entry.timestamp  # non-empty
    assert entry.context_refs == ["f.py"]
    assert entry.chosen_action == "action"


def test_query_by_task():
    dl = DecisionLog()
    dl.log_decision("a1", "t1", "code_change", [], [], "act1", "r1")
    dl.log_decision("a1", "t2", "code_change", [], [], "act2", "r2")
    dl.log_decision("a2", "t1", "file_selection", [], [], "act3", "r3")
    results = dl.query_by_task("t1")
    assert len(results) == 2


def test_query_by_agent():
    dl = DecisionLog()
    dl.log_decision("a1", "t1", "code_change", [], [], "act1", "r1")
    dl.log_decision("a2", "t1", "code_change", [], [], "act2", "r2")
    assert len(dl.query_by_agent("a1")) == 1


def test_query_by_type():
    dl = DecisionLog()
    dl.log_decision("a1", "t1", "code_change", [], [], "act1", "r1")
    dl.log_decision("a1", "t1", "escalation", [], [], "act2", "r2")
    assert len(dl.query_by_type("escalation")) == 1


def test_audit_trail_integration():
    audit = AuditTrail()
    dl = DecisionLog(audit_trail=audit)
    dl.log_decision("a1", "t1", "code_change", ["f.py"], [], "act", "reason")
    assert audit.total_entries == 1
    entries = audit.query_by_correlation("t1")
    assert len(entries) == 1
    assert entries[0].action_type == "decision"


def test_storage_persistence(tmp_path):
    from src.infrastructure.storage import FilesystemBackend
    storage = FilesystemBackend(base_path=str(tmp_path))
    dl = DecisionLog(storage=storage)
    dl.log_decision("a1", "t1", "code_change", [], [], "act", "reason")
    keys = storage.list_keys("decisions")
    assert len(keys) == 1
