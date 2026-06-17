"""Dry-run manager that intercepts side effects and produces simulation reports.

In dry-run mode, no actual file writes, PR creations, or commands are executed.
Instead, all operations are logged for review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DryRunReport:
    """Summary of what would happen if dry-run mode were disabled."""

    task_id: str
    files_modified: list[dict] = field(default_factory=list)
    prs_created: list[dict] = field(default_factory=list)
    work_items_created: list[dict] = field(default_factory=list)
    commands_executed: list[str] = field(default_factory=list)
    notifications_sent: list[dict] = field(default_factory=list)
    estimated_tokens: int = 0


class DryRunManager:
    """Intercepts side effects in dry-run mode, producing a simulation report."""

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self._file_writes: list[dict] = []
        self._pr_creations: list[dict] = []
        self._work_items: list[dict] = []
        self._commands: list[str] = []
        self._notifications: list[dict] = []
        self._token_estimate: int = 0

    @property
    def is_dry_run(self) -> bool:
        return self._enabled

    def log_file_write(self, path: str, diff: str) -> None:
        self._file_writes.append({"path": path, "diff_preview": diff[:500]})
        if self._enabled:
            logger.info("[DRY-RUN] Would write: %s", path)

    def log_pr_creation(self, repo: str, title: str, branch: str) -> None:
        self._pr_creations.append({"repo": repo, "title": title, "branch": branch})
        if self._enabled:
            logger.info("[DRY-RUN] Would create PR: %s on %s", title, repo)

    def log_work_item(self, type: str, title: str) -> None:
        self._work_items.append({"type": type, "title": title})
        if self._enabled:
            logger.info("[DRY-RUN] Would create %s: %s", type, title)

    def log_command(self, command: str) -> None:
        self._commands.append(command)
        if self._enabled:
            logger.info("[DRY-RUN] Would execute: %s", command)

    def log_notification(self, channel: str, message: str) -> None:
        self._notifications.append({"channel": channel, "message": message})

    def add_token_estimate(self, tokens: int) -> None:
        self._token_estimate += tokens

    def generate_report(self, task_id: str) -> DryRunReport:
        return DryRunReport(
            task_id=task_id,
            files_modified=list(self._file_writes),
            prs_created=list(self._pr_creations),
            work_items_created=list(self._work_items),
            commands_executed=list(self._commands),
            notifications_sent=list(self._notifications),
            estimated_tokens=self._token_estimate,
        )

    def reset(self) -> None:
        """Clear all logged operations."""
        self._file_writes.clear()
        self._pr_creations.clear()
        self._work_items.clear()
        self._commands.clear()
        self._notifications.clear()
        self._token_estimate = 0
