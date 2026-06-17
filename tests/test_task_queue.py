"""Tests for TaskQueue."""

import pytest

from src.core.enums import TaskPriority, TaskState
from src.infrastructure.task_queue import TaskQueue


def test_enqueue_and_dequeue():
    q = TaskQueue()
    tid = q.enqueue("Fix bug")
    task = q.dequeue()
    assert task is not None
    assert task.task_id == tid
    assert task.state == TaskState.IN_PROGRESS


def test_empty_dequeue():
    q = TaskQueue()
    assert q.dequeue() is None


# Property 23: Task Queue Priority Ordering
def test_priority_ordering():
    """Tasks dequeued in priority order: high > normal > low."""
    q = TaskQueue()
    low_id = q.enqueue("Low task", priority="low")
    normal_id = q.enqueue("Normal task", priority="normal")
    high_id = q.enqueue("High task", priority="high")

    t1 = q.dequeue()
    assert t1.task_id == high_id
    t2 = q.dequeue()
    assert t2.task_id == normal_id
    t3 = q.dequeue()
    assert t3.task_id == low_id


def test_fifo_within_same_priority():
    q = TaskQueue()
    id1 = q.enqueue("First")
    id2 = q.enqueue("Second")
    t1 = q.dequeue()
    assert t1.task_id == id1


# Property 24: Task Queue Workspace Conflict Prevention
def test_workspace_conflict():
    """Two tasks on same repo cannot both be in_progress."""
    q = TaskQueue()
    q.enqueue("Task A", repos_involved=["repo-1"])
    q.enqueue("Task B", repos_involved=["repo-1"])

    t1 = q.dequeue()
    assert t1 is not None
    t2 = q.dequeue()
    assert t2 is None  # Blocked by workspace lock


def test_workspace_conflict_released_after_complete():
    q = TaskQueue()
    id1 = q.enqueue("Task A", repos_involved=["repo-1"])
    q.enqueue("Task B", repos_involved=["repo-1"])

    t1 = q.dequeue()
    q.update_state(t1.task_id, "completed")

    t2 = q.dequeue()
    assert t2 is not None


# Property 25: Task Queue Dependency Enforcement
def test_dependency_enforcement():
    """Task with depends_on not dequeued until deps completed."""
    q = TaskQueue()
    dep_id = q.enqueue("Dependency task")
    q.enqueue("Dependent task", depends_on=[dep_id])

    # Dequeue dependency first
    t1 = q.dequeue()
    assert t1.task_id == dep_id

    # Dependent should not be available yet
    t2 = q.dequeue()
    assert t2 is None

    # Complete dependency
    q.update_state(dep_id, "completed")

    # Now dependent should be available
    t3 = q.dequeue()
    assert t3 is not None


def test_cancel_task():
    q = TaskQueue()
    tid = q.enqueue("Task to cancel")
    q.cancel(tid)
    task = q.get_task(tid)
    assert task.state == TaskState.CANCELLED


def test_list_tasks_by_state():
    q = TaskQueue()
    q.enqueue("Task 1")
    q.enqueue("Task 2")
    q.dequeue()  # moves one to in_progress
    queued = q.list_tasks(state="queued")
    in_progress = q.list_tasks(state="in_progress")
    assert len(queued) == 1
    assert len(in_progress) == 1


def test_failed_task_releases_locks():
    q = TaskQueue()
    q.enqueue("Task A", repos_involved=["repo-1"])
    q.enqueue("Task B", repos_involved=["repo-1"])

    t1 = q.dequeue()
    q.update_state(t1.task_id, "failed", reason="test failure")

    t2 = q.dequeue()
    assert t2 is not None


def test_get_task_not_found():
    q = TaskQueue()
    with pytest.raises(KeyError):
        q.get_task("nonexistent")


# Property 38: Task Queue State Persistence Round-Trip
def test_persistence(tmp_path):
    from src.infrastructure.storage import FilesystemBackend
    storage = FilesystemBackend(base_path=str(tmp_path))
    q = TaskQueue(storage=storage)
    tid = q.enqueue("Persisted task")
    data = storage.load("tasks", tid)
    assert data is not None
    assert data["description"] == "Persisted task"
