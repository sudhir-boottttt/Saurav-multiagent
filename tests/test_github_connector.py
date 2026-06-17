"""Property-based and unit tests for the GitHub Connector."""

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.integrations.github_connector import GitHubConnector, RateLimitError


# Feature: multi-agent-platform, Property 13: PR Creation Data Integrity


@given(
    title=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
    body=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
    branch=st.text(min_size=1, max_size=20, alphabet="abcdefghij-"),
)
@settings(max_examples=100)
def test_pr_creation_data_integrity(title, body, branch):
    """Created PR must have non-empty title, body, and branch."""
    gh = GitHubConnector(token="test-token")
    pr = gh.create_pr("https://github.com/org/repo", branch, title, body)
    assert pr.title == title
    assert pr.body == body
    assert pr.branch == branch
    assert pr.url  # non-empty
    assert pr.pr_number >= 1


# Feature: multi-agent-platform, Property 14: GitHub Rate Limit Handling


def test_rate_limit_blocks_requests():
    """Requests should be blocked when rate limited."""
    gh = GitHubConnector(token="test-token")
    gh.handle_rate_limit(time.time() + 60)  # rate limited for 60s
    with pytest.raises(RateLimitError):
        gh.clone_repo("https://github.com/org/repo")


def test_rate_limit_allows_after_reset():
    """Requests should succeed after rate limit resets (rate limit check passes)."""
    gh = GitHubConnector(token="test-token")
    gh.handle_rate_limit(time.time() - 1)  # already reset
    # After rate limit reset, the rate limit check should not raise.
    # We test with create_pr which doesn't need real git.
    pr = gh.create_pr("https://github.com/org/repo", "branch", "title", "body")
    assert pr.title == "title"


def test_queue_and_process_requests():
    """Queued requests should be processable after rate limit resets."""
    gh = GitHubConnector(token="test-token")
    gh.handle_rate_limit(time.time() + 60)
    gh.queue_request({"action": "clone", "repo": "test"})
    gh.queue_request({"action": "clone", "repo": "test2"})
    assert gh.process_queued_requests() == 0  # still rate limited
    gh.handle_rate_limit(time.time() - 1)  # reset
    assert gh.process_queued_requests() == 2


# --- Unit Tests ---


def test_pr_rejects_empty_title():
    gh = GitHubConnector(token="test-token")
    with pytest.raises(ValueError, match="title"):
        gh.create_pr("https://github.com/org/repo", "branch", "", "body")


def test_pr_rejects_empty_body():
    gh = GitHubConnector(token="test-token")
    with pytest.raises(ValueError, match="body"):
        gh.create_pr("https://github.com/org/repo", "branch", "title", "")


def test_get_file_diff():
    gh = GitHubConnector(token="test-token")
    diffs = gh.get_file_diff("https://github.com/org/repo", "abc123")
    assert len(diffs) >= 1
    assert diffs[0].file_path


def test_register_webhook():
    gh = GitHubConnector(token="test-token")
    wh_id = gh.register_webhook("https://github.com/org/repo", ["push", "pull_request"])
    assert wh_id  # non-empty UUID
