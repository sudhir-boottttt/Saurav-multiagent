"""Tests for SandboxExecutor."""

import pytest

from src.code_ops.sandbox import BlockedCommandError, CommandResult, SandboxExecutor


@pytest.fixture
def sandbox(tmp_path):
    return SandboxExecutor(base_path=str(tmp_path), max_timeout_seconds=10)


def test_execute_simple_command(sandbox):
    result = sandbox.execute("ws", "echo hello")
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.timed_out is False


# Property 28: Sandbox Structured Output
def test_structured_output(sandbox):
    """CommandResult has all required fields with correct types."""
    result = sandbox.execute("ws", "echo test")
    assert isinstance(result.exit_code, int)
    assert isinstance(result.stdout, str)
    assert isinstance(result.stderr, str)
    assert isinstance(result.duration_ms, int)
    assert result.duration_ms >= 0
    assert isinstance(result.command, str)
    assert isinstance(result.timed_out, bool)


def test_execute_failing_command(sandbox):
    result = sandbox.execute("ws", "false")
    assert result.exit_code != 0


# Property 27: Sandbox Timeout Enforcement
def test_timeout_enforcement(sandbox):
    """Command exceeding timeout should be killed with timed_out=True."""
    result = sandbox.execute("ws", "sleep 30", timeout=1)
    assert result.timed_out is True
    assert result.exit_code == -1


# Property 13: Verification Result Correctness
def test_exit_code_zero_means_passed(sandbox):
    result = sandbox.execute("ws", "true")
    assert result.exit_code == 0


def test_exit_code_nonzero_means_failed(sandbox):
    result = sandbox.execute("ws", "exit 1")
    assert result.exit_code != 0


def test_blocked_command_rm_rf(sandbox):
    with pytest.raises(BlockedCommandError):
        sandbox.execute("ws", "rm -rf /")


def test_blocked_command_sudo(sandbox):
    with pytest.raises(BlockedCommandError):
        sandbox.execute("ws", "sudo apt install something")


def test_blocked_command_shutdown(sandbox):
    with pytest.raises(BlockedCommandError):
        sandbox.execute("ws", "shutdown -h now")


def test_execute_sequence_stops_on_failure(sandbox):
    results = sandbox.execute_sequence("ws", [
        "echo step1",
        "false",
        "echo step3",
    ])
    assert len(results) == 2  # Stops after failure
    assert results[0].exit_code == 0
    assert results[1].exit_code != 0


def test_execute_sequence_all_pass(sandbox):
    results = sandbox.execute_sequence("ws", ["echo a", "echo b", "echo c"])
    assert len(results) == 3
    assert all(r.exit_code == 0 for r in results)


def test_workspace_path_traversal(sandbox):
    with pytest.raises(PermissionError):
        sandbox.execute("../../escape", "echo bad")


def test_captures_stderr(sandbox):
    result = sandbox.execute("ws", "echo error >&2")
    assert "error" in result.stderr
