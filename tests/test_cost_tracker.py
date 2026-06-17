"""Tests for CostTracker."""

from src.observability.cost_tracker import CostTracker


def test_log_usage():
    ct = CostTracker()
    ct.log_usage("agent-1", "task-1", 100, 50)
    assert ct.get_task_usage("task-1") == 150
    assert ct.get_agent_usage("agent-1") == 150


def test_task_budget_within():
    ct = CostTracker(max_tokens_per_task=1000)
    ct.log_usage("a", "t1", 400, 100)
    assert ct.check_task_budget("t1") is True


def test_task_budget_exceeded():
    ct = CostTracker(max_tokens_per_task=100)
    ct.log_usage("a", "t1", 60, 50)
    assert ct.check_task_budget("t1") is False


def test_daily_limit():
    ct = CostTracker(max_tokens_per_day=500)
    ct.log_usage("a", "t1", 200, 100)
    assert ct.check_daily_limit() is True
    ct.log_usage("a", "t2", 200, 100)
    assert ct.check_daily_limit() is False


# Property 19: Cost Tracking Consistency
def test_cost_tracking_consistency():
    """Sum of per-call tokens equals aggregate totals."""
    ct = CostTracker()
    ct.log_usage("agent-1", "task-1", 100, 50)
    ct.log_usage("agent-1", "task-1", 200, 100)
    ct.log_usage("agent-2", "task-1", 50, 25)

    assert ct.get_task_usage("task-1") == 100 + 50 + 200 + 100 + 50 + 25
    assert ct.get_agent_usage("agent-1") == 100 + 50 + 200 + 100
    assert ct.get_agent_usage("agent-2") == 50 + 25


def test_usage_summary():
    ct = CostTracker()
    ct.log_usage("a1", "t1", 100, 50)
    ct.log_usage("a2", "t2", 200, 100)
    summary = ct.get_usage_summary()
    assert summary["total_calls"] == 2
    assert summary["by_agent"]["a1"] == 150
    assert summary["by_task"]["t1"] == 150


def test_with_storage(tmp_path):
    from src.infrastructure.storage import FilesystemBackend
    storage = FilesystemBackend(base_path=str(tmp_path))
    ct = CostTracker(storage=storage)
    ct.log_usage("a1", "t1", 100, 50)
    keys = storage.list_keys("cost_records")
    assert len(keys) == 1
