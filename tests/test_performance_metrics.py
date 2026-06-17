"""Tests for AgentPerformanceMetrics and DryRunManager."""

from src.infrastructure.dry_run import DryRunManager
from src.observability.performance_metrics import AgentPerformanceMetrics


# Property 39: Performance Metrics Recording
def test_record_and_retrieve():
    m = AgentPerformanceMetrics()
    m.record_response_time("a1", 150.0)
    m.record_response_time("a1", 250.0)
    m.record_tool_call("a1", success=True)
    m.record_tool_call("a1", success=False)
    m.record_verification("a1", passed=True)

    summary = m.get_agent_summary("a1")
    assert summary["avg_response_time_ms"] == 200.0
    assert summary["tool_call_success_rate"] == 0.5
    assert summary["verification_pass_rate"] == 1.0


def test_task_outcome():
    m = AgentPerformanceMetrics()
    m.record_task_outcome("t1", "completed", 5000, 3, 10, 1)
    summary = m.get_platform_summary()
    assert summary["total_task_outcomes"] == 1


def test_alerts():
    m = AgentPerformanceMetrics(alert_verification_threshold=0.8)
    m.record_verification("a1", passed=False)
    m.record_verification("a1", passed=False)
    m.record_verification("a1", passed=True)
    alerts = m.check_alerts()
    assert len(alerts) == 1
    assert alerts[0]["agent_id"] == "a1"


def test_no_alerts_when_passing():
    m = AgentPerformanceMetrics(alert_verification_threshold=0.5)
    m.record_verification("a1", passed=True)
    m.record_verification("a1", passed=True)
    assert m.check_alerts() == []


# --- DryRunManager ---

def test_dry_run_disabled():
    dm = DryRunManager(enabled=False)
    assert dm.is_dry_run is False


def test_dry_run_enabled():
    dm = DryRunManager(enabled=True)
    assert dm.is_dry_run is True


# Property 31: Dry-Run File Immutability
def test_dry_run_logs_without_side_effects(tmp_path):
    """In dry-run mode, file writes are logged but not executed."""
    dm = DryRunManager(enabled=True)
    test_file = tmp_path / "test.txt"
    test_file.write_text("original")

    dm.log_file_write(str(test_file), "diff content")
    # File should remain unchanged
    assert test_file.read_text() == "original"


def test_generate_report():
    dm = DryRunManager(enabled=True)
    dm.log_file_write("src/main.py", "+new line")
    dm.log_pr_creation("repo", "Fix bug", "fix-branch")
    dm.log_work_item("Bug", "Fix login")
    dm.log_command("pytest")
    dm.add_token_estimate(1000)

    report = dm.generate_report("task-1")
    assert report.task_id == "task-1"
    assert len(report.files_modified) == 1
    assert len(report.prs_created) == 1
    assert len(report.work_items_created) == 1
    assert len(report.commands_executed) == 1
    assert report.estimated_tokens == 1000


def test_reset():
    dm = DryRunManager(enabled=True)
    dm.log_file_write("f.py", "diff")
    dm.reset()
    report = dm.generate_report("t1")
    assert len(report.files_modified) == 0
