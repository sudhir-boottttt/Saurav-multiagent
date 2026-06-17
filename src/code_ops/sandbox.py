"""Sandbox executor for running shell commands within agent workspaces.

Provides timeout enforcement, resource limits, blocked command validation,
and structured output.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_BLOCKED_COMMANDS = [
    "rm -rf /",
    "rm -rf /*",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=/dev/zero",
    ":(){:|:&};:",
    "chmod -R 777 /",
    "chown -R",
]


@dataclass
class CommandResult:
    """Structured result from a shell command execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    command: str
    timed_out: bool = False


class BlockedCommandError(PermissionError):
    """Raised when a blocked command is attempted."""

    def __init__(self, command: str, pattern: str):
        super().__init__(
            f"Command blocked: '{command}' matches blocked pattern '{pattern}'"
        )


class SandboxExecutor:
    """Runs shell commands within agent workspaces with timeout and resource limits."""

    def __init__(
        self,
        base_path: str = "/tmp/agent-repos/",
        max_timeout_seconds: int = 120,
        max_output_bytes: int = 1_048_576,
        blocked_commands: list[str] | None = None,
    ) -> None:
        self._base_path = Path(base_path).resolve()
        self._max_timeout = max_timeout_seconds
        self._max_output = max_output_bytes
        self._blocked = blocked_commands if blocked_commands is not None else DEFAULT_BLOCKED_COMMANDS

    def _validate_command(self, command: str) -> None:
        """Reject blocked commands."""
        cmd_lower = command.lower().strip()
        for pattern in self._blocked:
            if pattern.lower() in cmd_lower:
                raise BlockedCommandError(command, pattern)

    def _resolve_workspace(self, workspace: str) -> Path:
        """Resolve workspace path within base."""
        ws = (self._base_path / workspace).resolve()
        try:
            ws.relative_to(self._base_path)
        except ValueError:
            raise PermissionError(f"Workspace '{workspace}' escapes base path")
        return ws

    def execute(
        self,
        workspace: str,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        """Execute a single command in the workspace directory."""
        self._validate_command(command)
        ws_path = self._resolve_workspace(workspace)

        if not ws_path.exists():
            ws_path.mkdir(parents=True, exist_ok=True)

        effective_timeout = timeout or self._max_timeout
        start = time.monotonic()

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(ws_path),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            return CommandResult(
                exit_code=result.returncode,
                stdout=result.stdout[:self._max_output],
                stderr=result.stderr[:self._max_output],
                duration_ms=duration_ms,
                command=command,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start) * 1000)
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {effective_timeout}s",
                duration_ms=duration_ms,
                command=command,
                timed_out=True,
            )

    def execute_sequence(
        self,
        workspace: str,
        commands: list[str],
    ) -> list[CommandResult]:
        """Execute commands in sequence, stopping on first failure."""
        results = []
        for cmd in commands:
            result = self.execute(workspace, cmd)
            results.append(result)
            if result.exit_code != 0:
                break
        return results
