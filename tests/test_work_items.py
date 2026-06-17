"""Tests for WorkItemProvider and LocalWorkItemProvider."""

import pytest

from src.integrations.work_items import LocalWorkItemProvider, WorkItemProvider


def test_protocol_compliance(tmp_path):
    provider = LocalWorkItemProvider(base_path=str(tmp_path))
    assert isinstance(provider, WorkItemProvider)


# Property 8: WorkItemProvider Round-Trip
def test_create_and_query(tmp_path):
    """Create item then query returns same type, title, description."""
    p = LocalWorkItemProvider(base_path=str(tmp_path))
    wi = p.create_item("Bug", "Fix login", "Login page broken")
    results = p.query_items(str(wi.item_id))
    assert len(results) == 1
    assert results[0].type == "Bug"
    assert results[0].title == "Fix login"
    assert results[0].description == "Login page broken"


def test_create_child(tmp_path):
    p = LocalWorkItemProvider(base_path=str(tmp_path))
    parent = p.create_item("Feature", "Auth", "Authentication feature")
    child = p.create_child_item(parent.item_id, "Task", "Add OAuth")
    assert child.parent_id == parent.item_id


def test_create_child_missing_parent(tmp_path):
    p = LocalWorkItemProvider(base_path=str(tmp_path))
    with pytest.raises(KeyError):
        p.create_child_item(999, "Task", "Orphan")


def test_update_item(tmp_path):
    p = LocalWorkItemProvider(base_path=str(tmp_path))
    wi = p.create_item("Bug", "Fix it", "Broken")
    p.update_item(wi.item_id, {"state": "Active"})
    updated = p.query_items(str(wi.item_id))[0]
    assert updated.state == "Active"


def test_add_and_get_comments(tmp_path):
    p = LocalWorkItemProvider(base_path=str(tmp_path))
    wi = p.create_item("Bug", "Fix it", "Broken")
    c = p.add_comment(wi.item_id, "Working on it", "dev-1")
    assert c.text == "Working on it"
    comments = p.get_comments(wi.item_id)
    assert len(comments) == 1
    assert comments[0].author_agent_id == "dev-1"


def test_link_pr(tmp_path):
    p = LocalWorkItemProvider(base_path=str(tmp_path))
    wi = p.create_item("Bug", "Fix it", "Broken")
    updated = p.link_pr(wi.item_id, "https://github.com/org/repo/pull/1")
    assert updated.pr_url == "https://github.com/org/repo/pull/1"
    assert updated.state == "In Review"


def test_query_by_title(tmp_path):
    p = LocalWorkItemProvider(base_path=str(tmp_path))
    p.create_item("Bug", "Login broken", "desc")
    p.create_item("Feature", "Add search", "desc")
    results = p.query_items("login")
    assert len(results) == 1
    assert results[0].title == "Login broken"
