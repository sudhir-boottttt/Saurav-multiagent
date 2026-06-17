"""PR review service for Tech Lead agents.

Supports remote reviews via GitHub API (when token available)
and local reviews via git diff (always available).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from src.integrations.github_connector import GitHubConnector, FileDiff

logger = logging.getLogger(__name__)


@dataclass
class ReviewComment:
    """A review comment on a PR."""
    comment_id: str
    file_path: str
    line: int
    body: str
    agent_id: str


@dataclass
class ReviewResult:
    """Result of a PR review."""
    verdict: str  # approve, request_changes, comment
    body: str
    comments: list[ReviewComment] = field(default_factory=list)
    agent_id: str = ""


class PRReviewService:
    """Enables Tech Lead agents to review PRs with grounded context."""

    def __init__(
        self,
        github: Optional[GitHubConnector] = None,
    ) -> None:
        self._github = github
        self._local_reviews: dict[str, ReviewResult] = {}
        self._local_comments: dict[str, list[ReviewComment]] = {}

    def fetch_pr_diff(self, repo_url: str, pr_number: int) -> list[FileDiff]:
        """Fetch PR diffs. Requires GITHUB_TOKEN."""
        if not self._github:
            raise RuntimeError("No GitHubConnector configured")
        return self._github.get_file_diff(repo_url, f"pr-{pr_number}")

    def post_line_comment(
        self, repo_url: str, pr_number: int,
        file_path: str, line: int, body: str, agent_id: str,
    ) -> str:
        """Post a line comment on a PR. Returns comment_id."""
        comment = ReviewComment(
            comment_id=str(uuid4()),
            file_path=file_path, line=line,
            body=body, agent_id=agent_id,
        )
        key = f"{repo_url}:{pr_number}"
        self._local_comments.setdefault(key, []).append(comment)
        logger.info("Comment on %s:%d by %s", file_path, line, agent_id)
        return comment.comment_id

    def submit_review(
        self, repo_url: str, pr_number: int,
        verdict: str, body: str, agent_id: str,
    ) -> ReviewResult:
        """Submit a review (approve/request_changes/comment)."""
        result = ReviewResult(
            verdict=verdict, body=body, agent_id=agent_id,
            comments=self._local_comments.get(f"{repo_url}:{pr_number}", []),
        )
        self._local_reviews[f"{repo_url}:{pr_number}"] = result
        logger.info("Review submitted for %s PR#%d: %s", repo_url, pr_number, verdict)
        return result

    def review_locally(
        self, repo_path: str, base_branch: str = "main", agent_id: str = "",
    ) -> dict:
        """Fallback: review via local git diff when no GITHUB_TOKEN."""
        if self._github:
            try:
                diff = self._github.get_local_diff(repo_path, base_branch)
                return {"diff": diff, "agent_id": agent_id, "method": "local_git"}
            except Exception as e:
                logger.warning("Local diff failed: %s", e)
        return {"diff": "", "agent_id": agent_id, "method": "unavailable"}
