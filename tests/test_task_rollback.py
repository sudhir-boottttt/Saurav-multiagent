"""Tests for TaskRollback."""

from src.code_ops.task_rollback import TaskRollback


def test_disabled():
    tr = TaskRollback(enabled=False)
    assert tr.create_checkpoint("t1", "/tmp") == ""


def test_enabled_property():
    tr = TaskRollback(enabled=True)
    assert tr.enabled is True
    tr2 = TaskRollback(enabled=False)
    assert tr2.enabled is False


def test_rollback_no_checkpoint():
    tr = TaskRollback()
    import pytest
    with pytest.raises(ValueError, match="No checkpoint"):
        tr.rollback("nonexistent", "/tmp")
