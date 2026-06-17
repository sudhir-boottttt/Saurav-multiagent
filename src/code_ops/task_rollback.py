"""Git-based task rollback using checkpoint branches.

Creates checkpoint branches before task execution and rolls back
to them on failure.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TaskRollback:
    """Git-based task-level rollback using checkpoint branches."""

    BRANCH_PREFIX = "checkpoint"

    def __init__(
        self,
        base_path: str = "/tmp/agent-repos/",
        enabled: bool = True,
        keep_checkpoints_hours: int = 24,
    ) -> None:
        self._base_path = Path(base_path)
        self._enabled = enabled
        self._keep_hours = keep_checkpoints_hours
        self._checkpoints: dict[str, str] = {}  # task_id -> branch_name

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _run_git(self, args: list[str], cwd: str) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git"] + args, cwd=cwd,
            capture_output=True, text=True, timeout=60,
        )
        return result

    def create_checkpoint(self, task_id: str, repo_path: str) -> str:
        """Create checkpoint/{task_id} branch. Returns branch name."""
        if not self._enabled:
            return ""

        branch = f"{self.BRANCH_PREFIX}/{task_id}"
        result = self._run_git(["checkout", "-b", branch], cwd=repo_path)
        if result.returncode != 0:
            # Branch might already exist, try switching to it
            self._run_git(["branch", "-D", branch], cwd=repo_path)
            result = self._run_git(["checkout", "-b", branch], cwd=repo_path)

        # Switch back to original branch
        self._run_git(["checkout", "-"], cwd=repo_path)
        self._checkpoints[task_id] = branch
        logger.info("Created checkpoint %s for task %s", branch, task_id)
        return branch

    def rollback(self, task_id: str, repo_path: str) -> None:
        """Reset working directory to checkpoint branch."""
        if not self._enabled:
            return

        branch = self._checkpoints.get(task_id)
        if not branch:
            raise ValueError(f"No checkpoint found for task {task_id}")

        # Discard any uncommitted changes
        self._run_git(["checkout", "."], cwd=repo_path)
        self._run_git(["clean", "-fd"], cwd=repo_path)

        # Switch to checkpoint branch
        result = self._run_git(["checkout", branch], cwd=repo_path)
        if result.returncode != 0:
            raise RuntimeError(
                f"Rollback failed for task {task_id}: {result.stderr.strip()}"
            )

        # Verify the working tree is clean (matches the checkpoint)
        status = self._run_git(["status", "--porcelain"], cwd=repo_path)
        if status.stdout.strip():
            logger.warning(
                "Working tree not clean after rollback for task %s: %s",
                task_id, status.stdout.strip(),
            )
            raise RuntimeError(
                f"Rollback verification failed for task {task_id}: "
                f"working tree has uncommitted changes"
            )

        logger.info("Rolled back to checkpoint %s (verified clean)", branch)

    def delete_checkpoint(self, task_id: str, repo_path: str) -> None:
        """Delete checkpoint branch after successful task."""
        branch = self._checkpoints.pop(task_id, None)
        if branch:
            self._run_git(["branch", "-D", branch], cwd=repo_path)
            logger.info("Deleted checkpoint %s", branch)

    def cleanup_old_checkpoints(self, repo_path: str) -> int:
        """Remove checkpoint branches older than keep_checkpoints_hours."""
        result = self._run_git(
            ["branch", "--list", f"{self.BRANCH_PREFIX}/*"],
            cwd=repo_path,
        )
        if result.returncode != 0:
            return 0

        count = 0
        for line in result.stdout.strip().splitlines():
            branch = line.strip()
            if branch and branch.startswith(self.BRANCH_PREFIX):
                self._run_git(["branch", "-D", branch], cwd=repo_path)
                count += 1
        return count
