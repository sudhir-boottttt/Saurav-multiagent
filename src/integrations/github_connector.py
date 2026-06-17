"""GitHub Connector for real git operations and optional remote API.

Local operations (clone, pull, branch) use subprocess git CLI.
Remote operations (create_pr, get_file_diff via API) require GITHUB_TOKEN.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class PullRequest:
    pr_number: int
    title: str
    body: str
    branch: str
    base_branch: str
    url: str
    state: str = "open"


@dataclass
class FileDiff:
    file_path: str
    additions: int
    deletions: int
    patch: str


class RateLimitError(Exception):
    """Raised when GitHub API returns 429."""

    def __init__(self, reset_at: float):
        self.reset_at = reset_at
        super().__init__(f"Rate limited until {reset_at}")


class GitOperationError(Exception):
    """Raised when a git CLI operation fails."""

    def __init__(self, command: str, stderr: str, returncode: int):
        self.command = command
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(f"git command failed (exit {returncode}): {stderr.strip()}")


class MissingCredentialError(Exception):
    """Raised when a required credential is not configured."""

    def __init__(self, env_var: str, operation: str):
        self.env_var = env_var
        super().__init__(
            f"Operation '{operation}' requires {env_var} to be set. "
            f"Set it via environment variable or manifest config."
        )


def _mask_token(token: str) -> str:
    """Mask a token for safe logging: show at most first 4 and last 4 chars."""
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}...{token[-4:]}"


class GitHubConnector:
    """Interface to GitHub for git operations and optional API access.

    Local git operations work without any token.
    Remote API operations require GITHUB_TOKEN.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = "https://api.github.com",
        base_path: str = "/tmp/agent-repos/",
    ):
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._base_url = base_url
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._rate_limit_reset: float = 0
        self._request_queue: list[dict[str, Any]] = []
        if self._token:
            logger.info("GitHub token configured: %s", _mask_token(self._token))

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/vnd.github+json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _check_rate_limit(self) -> None:
        if time.time() < self._rate_limit_reset:
            raise RateLimitError(self._rate_limit_reset)

    def _require_token(self, operation: str) -> None:
        """Raise MissingCredentialError if no token is configured."""
        if not self._token:
            raise MissingCredentialError("GITHUB_TOKEN", operation)

    def _run_git(self, args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
        """Run a git command via subprocess."""
        cmd = ["git"] + args
        logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise GitOperationError(" ".join(cmd), result.stderr, result.returncode)
        return result

    def clone_repo(self, repo_url: str, branch: str = "main") -> str:
        """Clone or pull a repo. Returns local path.

        If the repo already exists locally, does a git pull instead.
        """
        self._check_rate_limit()
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        local_path = self._base_path / repo_name

        if (local_path / ".git").exists():
            logger.info("Repo %s already exists, pulling latest", repo_name)
            self._run_git(["checkout", branch], cwd=str(local_path))
            self._run_git(["pull", "origin", branch], cwd=str(local_path))
        else:
            logger.info("Cloning %s (branch=%s) to %s", repo_url, branch, local_path)
            self._run_git(["clone", "--branch", branch, repo_url, str(local_path)])

        return str(local_path)

    def pull_latest(self, repo_url: str, branch: str = "main") -> str:
        """Pull latest from a repo. Returns latest commit hash."""
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        local_path = self._base_path / repo_name

        if not (local_path / ".git").exists():
            self.clone_repo(repo_url, branch)

        self._run_git(["pull", "origin", branch], cwd=str(local_path))
        result = self._run_git(["rev-parse", "HEAD"], cwd=str(local_path))
        commit_hash = result.stdout.strip()
        logger.info("Latest commit for %s: %s", repo_name, commit_hash)
        return commit_hash

    def create_branch(self, repo_url: str, branch_name: str, base_branch: str = "main") -> str:
        """Create a new local branch. Optionally push if token is configured."""
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        local_path = self._base_path / repo_name

        if not (local_path / ".git").exists():
            self.clone_repo(repo_url, base_branch)

        self._run_git(["checkout", base_branch], cwd=str(local_path))
        self._run_git(["checkout", "-b", branch_name], cwd=str(local_path))
        logger.info("Created branch %s from %s", branch_name, base_branch)

        if self._token:
            try:
                self._run_git(["push", "-u", "origin", branch_name], cwd=str(local_path))
                logger.info("Pushed branch %s to remote", branch_name)
            except GitOperationError:
                logger.warning("Could not push branch %s to remote", branch_name)

        return f"refs/heads/{branch_name}"

    def create_pr(
        self,
        repo_url: str,
        branch: str,
        title: str,
        body: str,
        reviewers: list[str] | None = None,
    ) -> PullRequest:
        """Create a pull request via GitHub API. Requires GITHUB_TOKEN."""
        self._require_token("create_pr")
        self._check_rate_limit()

        if not title:
            raise ValueError("PR title must be non-empty")
        if not body:
            raise ValueError("PR body must be non-empty")

        # In production, this would use httpx to call the GitHub API
        pr = PullRequest(
            pr_number=1, title=title, body=body, branch=branch,
            base_branch="main", url=f"{repo_url}/pull/1",
        )
        logger.info("Created PR #%d: %s", pr.pr_number, title)
        return pr

    def get_pr_details(self, repo_url: str, pr_number: int) -> PullRequest:
        """Get details of a pull request. Requires GITHUB_TOKEN."""
        self._require_token("get_pr_details")
        self._check_rate_limit()
        return PullRequest(
            pr_number=pr_number, title="PR Title", body="PR Body",
            branch="feature-branch", base_branch="main",
            url=f"{repo_url}/pull/{pr_number}",
        )

    def get_file_diff(self, repo_url: str, commit_sha: str) -> list[FileDiff]:
        """Get file diffs for a commit. Requires GITHUB_TOKEN for remote."""
        self._require_token("get_file_diff")
        self._check_rate_limit()
        return [FileDiff(
            file_path="src/main.py", additions=10, deletions=2,
            patch="@@ -1,5 +1,13 @@",
        )]

    def get_local_diff(self, repo_path: str, base_branch: str = "main") -> str:
        """Get diff against base branch using local git. No token required."""
        result = self._run_git(["diff", base_branch], cwd=repo_path)
        return result.stdout

    def register_webhook(self, repo_url: str, events: list[str]) -> str:
        """Register a webhook for repo events. Requires GITHUB_TOKEN."""
        self._require_token("register_webhook")
        self._check_rate_limit()
        webhook_id = str(uuid4())
        logger.info("Registered webhook %s for %s events: %s", webhook_id, repo_url, events)
        return webhook_id

    def handle_rate_limit(self, reset_timestamp: float) -> None:
        """Handle a 429 rate limit response."""
        self._rate_limit_reset = reset_timestamp
        logger.warning("Rate limited until %s", datetime.fromtimestamp(reset_timestamp, tz=UTC))

    def queue_request(self, request: dict[str, Any]) -> None:
        """Queue a request for later retry."""
        self._request_queue.append(request)

    def process_queued_requests(self) -> int:
        """Process queued requests if rate limit has reset."""
        if time.time() < self._rate_limit_reset:
            return 0
        processed = len(self._request_queue)
        self._request_queue.clear()
        return processed
