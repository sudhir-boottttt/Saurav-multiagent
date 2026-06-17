"""Tests for new enums and data models added in real-data-sources."""

import pytest
from pydantic import ValidationError

from src.core.enums import (
    HIPUrgency,
    RepoStatus,
    ReviewVerdict,
    TaskPriority,
    TaskState,
)
from src.core.models import (
    CostRecord,
    HIPRecord,
    RepoState,
    RuntimeConfig,
    TaskEntry,
    ToolDefinition,
    VerificationResult,
)


# --- Enum tests ---

def test_task_state_values():
    assert TaskState.QUEUED == "queued"
    assert TaskState.IN_PROGRESS == "in_progress"
    assert TaskState.COMPLETED == "completed"
    assert TaskState.FAILED == "failed"
    assert TaskState.STALLED == "stalled"


def test_task_priority_values():
    assert TaskPriority.HIGH == "high"
    assert TaskPriority.NORMAL == "normal"
    assert TaskPriority.LOW == "low"


def test_review_verdict_values():
    assert ReviewVerdict.APPROVE == "approve"
    assert ReviewVerdict.REQUEST_CHANGES == "request_changes"


def test_hip_urgency_values():
    assert HIPUrgency.LOW == "low"
    assert HIPUrgency.HIGH == "high"


def test_repo_status_values():
    assert RepoStatus.CLONED == "cloned"
    assert RepoStatus.READY == "ready"
    assert RepoStatus.ERROR == "error"


# --- Model tests ---

def test_task_entry_defaults():
    task = TaskEntry(description="Fix bug")
    assert task.state == TaskState.QUEUED
    assert task.priority == TaskPriority.NORMAL
    assert task.task_id  # auto-generated
    assert task.depends_on == []


def test_task_entry_rejects_empty_description():
    with pytest.raises(ValidationError):
        TaskEntry(description="")


def test_tool_definition():
    tool = ToolDefinition(name="search_code", description="Search code")
    assert tool.name == "search_code"
    assert tool.parameters_schema == {}


def test_tool_definition_rejects_empty_name():
    with pytest.raises(ValidationError):
        ToolDefinition(name="")


def test_verification_result():
    vr = VerificationResult(passed=True, command_results=[{"cmd": "pytest", "exit_code": 0}])
    assert vr.passed is True


def test_repo_state_defaults():
    rs = RepoState(repo_name="my-service")
    assert rs.status == RepoStatus.CLONED
    assert rs.branch == "main"
    assert rs.language == "python"


def test_runtime_config():
    rc = RuntimeConfig(language="python", test_command="pytest")
    assert rc.language == "python"
    assert rc.test_command == "pytest"


def test_cost_record():
    cr = CostRecord(agent_id="agent-1", input_tokens=100, output_tokens=50)
    assert cr.total_tokens == 0  # Not auto-computed, just a field
    assert cr.record_id  # auto-generated


def test_cost_record_rejects_negative_tokens():
    with pytest.raises(ValidationError):
        CostRecord(agent_id="a", input_tokens=-1)


def test_hip_record():
    hip = HIPRecord(question="Approve this?", agent_id="agent-1")
    assert hip.urgency == HIPUrgency.NORMAL
    assert hip.response is None
    assert hip.hip_id  # auto-generated


def test_hip_record_rejects_empty_question():
    with pytest.raises(ValidationError):
        HIPRecord(question="", agent_id="agent-1")
