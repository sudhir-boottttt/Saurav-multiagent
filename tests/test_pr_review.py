"""Tests for PRReviewService."""

from src.integrations.pr_review import PRReviewService


def test_post_line_comment():
    svc = PRReviewService()
    cid = svc.post_line_comment("repo", 1, "src/main.py", 10, "Fix this", "tl-1")
    assert cid  # non-empty


def test_submit_review():
    svc = PRReviewService()
    svc.post_line_comment("repo", 1, "src/main.py", 10, "Issue here", "tl-1")
    result = svc.submit_review("repo", 1, "request_changes", "Needs fixes", "tl-1")
    assert result.verdict == "request_changes"
    assert len(result.comments) == 1


def test_review_locally_no_github():
    svc = PRReviewService(github=None)
    result = svc.review_locally("/tmp/repo")
    assert result["method"] == "unavailable"


def test_submit_approve():
    svc = PRReviewService()
    result = svc.submit_review("repo", 1, "approve", "LGTM", "tl-1")
    assert result.verdict == "approve"
