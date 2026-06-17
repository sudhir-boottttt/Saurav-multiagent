"""Local file manager with workspace isolation and path traversal protection.

Manages file read/write/patch within agent workspaces. Provides backup/rollback,
file-level locking, and verification via shell commands.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.models import VerificationResult

logger = logging.getLogger(__name__)


class PathTraversalError(PermissionError):
    """Raised when a path escapes the workspace boundary."""

    def __init__(self, path: str, workspace: str):
        super().__init__(
            f"Path traversal blocked: '{path}' escapes workspace '{workspace}'"
        )


class FileLockConflictError(Exception):
    """Raised when a file is already locked by another agent."""

    def __init__(self, file_path: str, locked_by: str):
        self.locked_by = locked_by
        super().__init__(
            f"File '{file_path}' is locked by agent '{locked_by}'"
        )


@dataclass
class _FileLock:
    agent_id: str
    acquired_at: float = field(default_factory=time.monotonic)


class LocalFileManager:
    """Manages file read/write/patch within agent workspaces with path traversal protection."""

    # Roles that are read-only (cannot write files)
    READ_ONLY_ROLES = {"pm_agent", "tech_lead_agent"}

    def __init__(self, base_path: str = "/tmp/agent-repos/") -> None:
        self._base_path = Path(base_path).resolve()
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._backup_dir = self._base_path / ".backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, _FileLock] = {}
        self._lock_mutex = threading.Lock()

    def _resolve_safe_path(self, workspace: str, file_path: str) -> Path:
        """Resolve path and reject traversal attempts."""
        ws_path = (self._base_path / workspace).resolve()

        # Reject absolute paths outside workspace
        if Path(file_path).is_absolute():
            candidate = Path(file_path).resolve()
        else:
            candidate = (ws_path / file_path).resolve()

        # Check the resolved path is within workspace
        try:
            candidate.relative_to(ws_path)
        except ValueError:
            raise PathTraversalError(file_path, workspace)

        # Check for symlinks escaping workspace
        if candidate.is_symlink():
            real = candidate.resolve()
            try:
                real.relative_to(ws_path)
            except ValueError:
                raise PathTraversalError(file_path, workspace)

        return candidate

    def _check_write_permission(
        self, workspace: str, agent_role: Optional[str],
    ) -> None:
        """Raise PermissionError if agent role is read-only."""
        if agent_role and agent_role in self.READ_ONLY_ROLES:
            raise PermissionError(
                f"Agent role '{agent_role}' does not have write access to workspace '{workspace}'"
            )

    def _check_repo_access(
        self, workspace: str, agent_role: Optional[str],
        assigned_repos: Optional[list[str]],
    ) -> None:
        """Raise PermissionError if dev agent accesses a repo not in assigned_repos."""
        if (
            agent_role == "developer_agent"
            and assigned_repos is not None
            and workspace not in assigned_repos
        ):
            raise PermissionError(
                f"Developer agent not assigned to workspace '{workspace}'. "
                f"Assigned repos: {assigned_repos}"
            )

    def read_file(
        self, workspace: str, file_path: str,
        agent_role: Optional[str] = None,
        assigned_repos: Optional[list[str]] = None,
    ) -> str:
        """Read file contents. Raises PermissionError if path escapes workspace or repo not assigned."""
        self._check_repo_access(workspace, agent_role, assigned_repos)
        safe_path = self._resolve_safe_path(workspace, file_path)
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: {file_path} in workspace {workspace}")
        return safe_path.read_text(encoding="utf-8")

    def write_file(
        self, workspace: str, file_path: str, content: str,
        agent_role: Optional[str] = None,
        assigned_repos: Optional[list[str]] = None,
    ) -> None:
        """Write file contents with backup. Raises PermissionError on path traversal or role violation."""
        self._check_repo_access(workspace, agent_role, assigned_repos)
        self._check_write_permission(workspace, agent_role)
        safe_path = self._resolve_safe_path(workspace, file_path)

        # Create backup if file exists
        if safe_path.exists():
            self._create_backup(workspace, file_path, safe_path)

        # Ensure parent directory exists
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        logger.info("Wrote file %s in workspace %s", file_path, workspace)

    def apply_patch(self, workspace: str, file_path: str, patch: str) -> None:
        """Apply a unified diff patch to a file."""
        safe_path = self._resolve_safe_path(workspace, file_path)
        if not safe_path.exists():
            raise FileNotFoundError(f"Cannot patch non-existent file: {file_path}")

        # Backup before patching
        self._create_backup(workspace, file_path, safe_path)

        # Write patch to temp file and apply
        ws_path = (self._base_path / workspace).resolve()
        patch_file = ws_path / ".tmp_patch"
        try:
            patch_file.write_text(patch, encoding="utf-8")
            result = subprocess.run(
                ["patch", str(safe_path), str(patch_file)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Patch failed: {result.stderr}")
        finally:
            patch_file.unlink(missing_ok=True)

    def rollback_file(self, workspace: str, file_path: str) -> None:
        """Restore file from backup."""
        backup_path = self._backup_path(workspace, file_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"No backup found for {file_path} in workspace {workspace}")

        safe_path = self._resolve_safe_path(workspace, file_path)
        shutil.copy2(str(backup_path), str(safe_path))
        logger.info("Rolled back %s in workspace %s", file_path, workspace)

    def verify_change(self, workspace: str, validation_commands: list[str]) -> VerificationResult:
        """Run validation commands and return structured result."""
        ws_path = (self._base_path / workspace).resolve()
        command_results = []
        all_passed = True

        for cmd in validation_commands:
            try:
                result = subprocess.run(
                    cmd, shell=True, cwd=str(ws_path),
                    capture_output=True, text=True, timeout=120,
                )
                passed = result.returncode == 0
                command_results.append({
                    "command": cmd,
                    "exit_code": result.returncode,
                    "stdout": result.stdout[:4096],
                    "stderr": result.stderr[:4096],
                    "duration_ms": 0,
                })
                if not passed:
                    all_passed = False
            except subprocess.TimeoutExpired:
                command_results.append({
                    "command": cmd,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": "Command timed out",
                    "duration_ms": 120000,
                })
                all_passed = False

        return VerificationResult(passed=all_passed, command_results=command_results)

    def acquire_file_lock(self, workspace: str, file_path: str, agent_id: str) -> bool:
        """Acquire file-level lock for concurrent write protection."""
        lock_key = f"{workspace}:{file_path}"
        with self._lock_mutex:
            existing = self._locks.get(lock_key)
            if existing and existing.agent_id != agent_id:
                raise FileLockConflictError(file_path, existing.agent_id)
            self._locks[lock_key] = _FileLock(agent_id=agent_id)
            return True

    def release_file_lock(self, workspace: str, file_path: str, agent_id: str) -> None:
        """Release file-level lock."""
        lock_key = f"{workspace}:{file_path}"
        with self._lock_mutex:
            existing = self._locks.get(lock_key)
            if existing and existing.agent_id == agent_id:
                del self._locks[lock_key]

    def _create_backup(self, workspace: str, file_path: str, safe_path: Path) -> None:
        """Create a backup of a file before modification."""
        backup = self._backup_path(workspace, file_path)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(safe_path), str(backup))

    def _backup_path(self, workspace: str, file_path: str) -> Path:
        """Get the backup path for a file."""
        # Flatten the file path for backup storage
        flat_name = file_path.replace("/", "__").replace("\\", "__")
        return self._backup_dir / workspace / flat_name
