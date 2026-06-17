"""Property-based and unit tests for the Azure DevOps Connector."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.integrations.devops_connector import DevOpsConnector


# Feature: multi-agent-platform, Property 15: Work Item Field Completeness


@given(
    title=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
    desc=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N"))),
    priority=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=100)
def test_work_item_field_completeness(title, desc, priority):
    """Created work items must have title, description, and priority."""
    ado = DevOpsConnector()
    wi = ado.create_work_item("Feature", title, desc, {"priority": priority})
    assert wi.title == title
    assert wi.description == desc
    assert wi.priority == priority
    assert wi.item_id >= 1


# Feature: multi-agent-platform, Property 16: Work Item Status Sync


def test_pr_link_sets_in_review():
    """Linking a PR should set work item state to In Review."""
    ado = DevOpsConnector()
    wi = ado.create_work_item("User Story", "Story", "Description")
    ado.link_pr_to_work_item(wi.item_id, "https://github.com/org/repo/pull/1")
    updated = ado.get_work_item(wi.item_id)
    assert updated.state == "In Review"
    assert updated.pr_url == "https://github.com/org/repo/pull/1"


def test_status_change_to_done():
    """Status change webhook should update work item state."""
    ado = DevOpsConnector()
    wi = ado.create_work_item("User Story", "Story", "Description")
    ado.handle_status_change(wi.item_id, "Done")
    assert ado.get_work_item(wi.item_id).state == "Done"


# --- Unit Tests ---


def test_create_child_work_item():
    ado = DevOpsConnector()
    parent = ado.create_work_item("Feature", "Parent Feature", "Description")
    child = ado.create_child_work_item(parent.item_id, "User Story", "Child Story")
    assert child.parent_id == parent.item_id


def test_create_child_invalid_parent_raises():
    ado = DevOpsConnector()
    with pytest.raises(KeyError, match="not found"):
        ado.create_child_work_item(999, "User Story", "Orphan")


def test_work_item_rejects_empty_title():
    ado = DevOpsConnector()
    with pytest.raises(ValueError, match="title"):
        ado.create_work_item("Feature", "", "desc")


def test_work_item_rejects_empty_description():
    ado = DevOpsConnector()
    with pytest.raises(ValueError, match="description"):
        ado.create_work_item("Feature", "title", "")


def test_update_work_item():
    ado = DevOpsConnector()
    wi = ado.create_work_item("Bug", "Bug Title", "Bug Desc")
    ado.update_work_item(wi.item_id, {"state": "Active", "priority": 1})
    updated = ado.get_work_item(wi.item_id)
    assert updated.state == "Active"
    assert updated.priority == 1


def test_register_webhook():
    ado = DevOpsConnector()
    wh_id = ado.register_webhook(["workitem.updated"])
    assert wh_id
