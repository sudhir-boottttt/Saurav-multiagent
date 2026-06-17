"""Priority task queue with workspace locking and dependency enforcement.

Manages task lifecycle: queued → in_progress → waiting_on_human → completed/failed.
Persists state via StorageBackend.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from src.core.enums import TaskPriority, TaskState
from src.core.models import TaskEntry

if TYPE_CHECKING:
    from src.infrastructure.storage import StorageBackend

logger = logging.getLogger(__name__)

_PRIORITY_ORDER = {TaskPriority.HIGH: 0, TaskPriority.NORMAL: 1, TaskPriority.LOW: 2}


class TaskQueue:
    """Priority queue for task management with workspace locking."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        max_concurrent: int = 3,
    ) -> None:
        self._storage = storage
        self._max_concurrent = max_concurrent
        self._tasks: dict[str, TaskEntry] = {}
        self._workspace_locks: dict[str, str] = {}  # repo -> task_id

    def enqueue(
        self,
        description: str,
        priority: str = "normal",
        depends_on: list[str] | None = None,
        repos_involved: list[str] | None = None,
    ) -> str:
        """Add task to queue. Returns task_id."""
        task = TaskEntry(
            task_id=str(uuid4()),
            description=description,
            priority=TaskPriority(priority),
            state=TaskState.QUEUED,
            depends_on=depends_on or [],
            repos_involved=repos_involved or [],
        )
        self._tasks[task.task_id] = task
        self._persist(task)
        logger.info("Enqueued task %s: %s (priority=%s)", task.task_id, description, priority)
        return task.task_id

    def dequeue(self) -> Optional[TaskEntry]:
        """Get next eligible task (respecting priority, deps, workspace locks)."""
        eligible = []
        for task in self._tasks.values():
            if task.state != TaskState.QUEUED:
                continue
            if not self._deps_satisfied(task):
                continue
            if self.check_workspace_conflict(task.repos_involved):
                continue
            eligible.append(task)

        if not eligible:
            return None

        # Sort by priority then creation time
        eligible.sort(key=lambda t: (_PRIORITY_ORDER.get(t.priority, 1), t.created_at))
        task = eligible[0]

        # Acquire workspace locks
        for repo in task.repos_involved:
            self._workspace_locks[repo] = task.task_id

        task.state = TaskState.IN_PROGRESS
        task.started_at = datetime.now(UTC)
        self._persist(task)
        return task

    def update_state(
        self,
        task_id: str,
        state: str,
        reason: str | None = None,
    ) -> None:
        """Update task state."""
        task = self._get_task(task_id)
        task.state = TaskState(state)

        if state == "completed":
            task.completed_at = datetime.now(UTC)
            self._release_locks(task_id)
        elif state == "failed":
            task.completed_at = datetime.now(UTC)
            task.failure_reason = reason
            self._release_locks(task_id)
        elif state == "cancelled":
            self._release_locks(task_id)

        self._persist(task)

    def cancel(self, task_id: str) -> None:
        """Cancel a task."""
        self.update_state(task_id, "cancelled")

    def list_tasks(self, state: str | None = None) -> list[TaskEntry]:
        """List tasks, optionally filtered by state."""
        tasks = list(self._tasks.values())
        if state:
            tasks = [t for t in tasks if t.state == TaskState(state)]
        return tasks

    def check_workspace_conflict(self, repos: list[str]) -> bool:
        """Check if any repo is locked by an in-progress task."""
        for repo in repos:
            if repo in self._workspace_locks:
                return True
        return False

    def get_task(self, task_id: str) -> TaskEntry:
        """Get a task by ID."""
        return self._get_task(task_id)

    def _get_task(self, task_id: str) -> TaskEntry:
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found")
        return self._tasks[task_id]

    def _deps_satisfied(self, task: TaskEntry) -> bool:
        """Check if all dependencies are completed."""
        for dep_id in task.depends_on:
            dep = self._tasks.get(dep_id)
            if not dep or dep.state != TaskState.COMPLETED:
                return False
        return True

    def _release_locks(self, task_id: str) -> None:
        """Release workspace locks held by a task."""
        to_remove = [repo for repo, tid in self._workspace_locks.items() if tid == task_id]
        for repo in to_remove:
            del self._workspace_locks[repo]

    def _persist(self, task: TaskEntry) -> None:
        if self._storage:
            self._storage.save("tasks", task.task_id, task.model_dump(mode="json"))
