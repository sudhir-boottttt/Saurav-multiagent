"""Tests for LocalFileManager with workspace isolation."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.code_ops.local_file_manager import (
    FileLockConflictError,
    LocalFileManager,
    PathTraversalError,
)


@pytest.fixture
def lfm(tmp_path):
    mgr = LocalFileManager(base_path=str(tmp_path))
    # Create a workspace with a test file
    ws = tmp_path / "test-repo"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "main.py").write_text("print('hello')\n")
    (ws / "README.md").write_text("# Test\n")
    return mgr


def test_read_file(lfm):
    content = lfm.read_file("test-repo", "src/main.py")
    assert content == "print('hello')\n"


def test_read_file_not_found(lfm):
    with pytest.raises(FileNotFoundError):
        lfm.read_file("test-repo", "nonexistent.py")


def test_write_file(lfm):
    lfm.write_file("test-repo", "new_file.py", "x = 1\n")
    content = lfm.read_file("test-repo", "new_file.py")
    assert content == "x = 1\n"


def test_write_creates_backup(lfm):
    original = lfm.read_file("test-repo", "src/main.py")
    lfm.write_file("test-repo", "src/main.py", "print('modified')\n")
    # Rollback should restore original
    lfm.rollback_file("test-repo", "src/main.py")
    assert lfm.read_file("test-repo", "src/main.py") == original


# Property 10: Path Traversal Protection
@given(
    traversal=st.sampled_from([
        "../../../etc/passwd",
        "../../secret",
        "../outside",
        "src/../../escape",
    ])
)
@settings(max_examples=10)
def test_path_traversal_blocked(traversal):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mgr = LocalFileManager(base_path=tmp)
        import os
        os.makedirs(os.path.join(tmp, "ws"), exist_ok=True)
        with pytest.raises(PermissionError):
            mgr.read_file("ws", traversal)


def test_absolute_path_outside_blocked(lfm):
    with pytest.raises(PermissionError):
        lfm.read_file("test-repo", "/etc/passwd")


# Property 12: File Backup and Rollback
@given(content=st.text(min_size=1, max_size=500))
@settings(max_examples=20)
def test_backup_and_rollback(content):
    """Backup then rollback restores exact original content."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mgr = LocalFileManager(base_path=tmp)
        import os
        ws = os.path.join(tmp, "ws")
        os.makedirs(ws, exist_ok=True)
        with open(os.path.join(ws, "file.txt"), "w") as f:
            f.write("original content")

        original = mgr.read_file("ws", "file.txt")
        mgr.write_file("ws", "file.txt", content)  # triggers backup
        mgr.rollback_file("ws", "file.txt")
        assert mgr.read_file("ws", "file.txt") == original


def test_rollback_no_backup(lfm):
    with pytest.raises(FileNotFoundError, match="No backup"):
        lfm.rollback_file("test-repo", "no_backup.py")


# Property 37: File Locking for Concurrent Writes
def test_file_lock_acquired(lfm):
    assert lfm.acquire_file_lock("test-repo", "src/main.py", "agent-1") is True


def test_file_lock_conflict(lfm):
    lfm.acquire_file_lock("test-repo", "src/main.py", "agent-1")
    with pytest.raises(FileLockConflictError, match="agent-1"):
        lfm.acquire_file_lock("test-repo", "src/main.py", "agent-2")


def test_file_lock_same_agent(lfm):
    lfm.acquire_file_lock("test-repo", "src/main.py", "agent-1")
    # Same agent can re-acquire
    assert lfm.acquire_file_lock("test-repo", "src/main.py", "agent-1") is True


def test_file_lock_release(lfm):
    lfm.acquire_file_lock("test-repo", "src/main.py", "agent-1")
    lfm.release_file_lock("test-repo", "src/main.py", "agent-1")
    # Now another agent can acquire
    assert lfm.acquire_file_lock("test-repo", "src/main.py", "agent-2") is True


def test_verify_change_pass(lfm):
    result = lfm.verify_change("test-repo", ["echo 'ok'"])
    assert result.passed is True
    assert result.command_results[0]["exit_code"] == 0


def test_verify_change_fail(lfm):
    result = lfm.verify_change("test-repo", ["false"])
    assert result.passed is False
    assert result.command_results[0]["exit_code"] != 0


def test_write_nested_directory(lfm):
    lfm.write_file("test-repo", "deep/nested/dir/file.py", "x = 1\n")
    assert lfm.read_file("test-repo", "deep/nested/dir/file.py") == "x = 1\n"
