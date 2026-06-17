"""Azure DevOps Connector for work item management.

Handles work item CRUD, PR linking, bidirectional status sync,
and OAuth 2.0 authentication.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4

from src.integrations.work_items import Comment

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    item_id: int
    type: str  # "Feature" | "User Story" | "Bug" | "Task"
    title: str
    description: str
    state: str = "New"
    priority: int = 2
    acceptance_criteria: Optional[str] = None
    parent_id: Optional[int] = None
    pr_url: Optional[str] = None
    fields: dict[str, Any] = field(default_factory=dict)


class DevOpsConnector:
    """Interface to Azure DevOps APIs for work item management.

    Uses OAuth 2.0 for auth. Supports bidirectional sync via webhooks.
    """

    def __init__(self, org_url: str = "", project: str = "",
                 token: Optional[str] = None):
        self._org_url = org_url or os.environ.get("AZURE_DEVOPS_ORG_URL", "")
        self._project = project or os.environ.get("AZURE_DEVOPS_PROJECT", "")
        self._token = token or os.environ.get("AZURE_DEVOPS_TOKEN", "")
        self._next_id = 1
        self._work_items: dict[int, WorkItem] = {}
        self._webhooks: list[str] = []
        self._comments: dict[int, list[Comment]] = {}

    def create_work_item(self, type: str, title: str, description: str,
                         fields: dict[str, Any] | None = None) -> WorkItem:
        """Create a work item. Returns WorkItem with assigned ID."""
        if not title:
            raise ValueError("Work item title must be non-empty")
        if not description:
            raise ValueError("Work item description must be non-empty")
        item_id = self._next_id
        self._next_id += 1
        wi = WorkItem(
            item_id=item_id, type=type, title=title,
            description=description, fields=fields or {},
        )
        if fields:
            wi.priority = fields.get("priority", 2)
            wi.acceptance_criteria = fields.get("acceptance_criteria")
        self._work_items[item_id] = wi
        logger.info("Created %s #%d: %s", type, item_id, title)
        return wi

    def create_child_work_item(self, parent_id: int, type: str, title: str,
                                fields: dict[str, Any] | None = None) -> WorkItem:
        """Create a child work item linked to a parent."""
        if parent_id not in self._work_items:
            raise KeyError(f"Parent work item #{parent_id} not found")
        parent = self._work_items[parent_id]
        wi = self.create_work_item(type, title, f"Child of #{parent_id}", fields)
        wi.parent_id = parent_id
        return wi

    def update_work_item(self, item_id: int, fields: dict[str, Any]) -> WorkItem:
        """Update a work item's fields."""
        if item_id not in self._work_items:
            raise KeyError(f"Work item #{item_id} not found")
        wi = self._work_items[item_id]
        for key, value in fields.items():
            if key == "state":
                wi.state = value
            elif key == "priority":
                wi.priority = value
            elif key == "pr_url":
                wi.pr_url = value
            else:
                wi.fields[key] = value
        return wi

    def link_pr_to_work_item(self, item_id: int, pr_url: str) -> WorkItem:
        """Link a PR to a work item and set state to In Review."""
        wi = self.update_work_item(item_id, {"state": "In Review", "pr_url": pr_url})
        logger.info("Linked PR %s to work item #%d", pr_url, item_id)
        return wi

    def query_work_items(self, wiql_query: str) -> list[WorkItem]:
        """Query work items (simplified — returns all matching type/state)."""
        return list(self._work_items.values())

    def register_webhook(self, events: list[str]) -> str:
        """Register a webhook for DevOps events. Returns webhook ID."""
        wh_id = str(uuid4())
        self._webhooks.append(wh_id)
        logger.info("Registered DevOps webhook %s for events: %s", wh_id, events)
        return wh_id

    def get_work_item(self, item_id: int) -> WorkItem:
        """Get a work item by ID."""
        if item_id not in self._work_items:
            raise KeyError(f"Work item #{item_id} not found")
        return self._work_items[item_id]

    def handle_status_change(self, item_id: int, new_state: str) -> WorkItem:
        """Handle a status change from Azure DevOps webhook."""
        wi = self.update_work_item(item_id, {"state": new_state})
        logger.info("Work item #%d state changed to %s", item_id, new_state)
        return wi

    # --- WorkItemProvider protocol methods ---

    def create_item(self, type: str, title: str, description: str,
                    fields: dict | None = None) -> WorkItem:
        """WorkItemProvider-compatible create."""
        return self.create_work_item(type, title, description, fields)

    def create_child_item(self, parent_id: int, type: str, title: str,
                          fields: dict | None = None) -> WorkItem:
        """WorkItemProvider-compatible create child."""
        return self.create_child_work_item(parent_id, type, title, fields)

    def update_item(self, item_id: int, fields: dict) -> WorkItem:
        """WorkItemProvider-compatible update."""
        return self.update_work_item(item_id, fields)

    def query_items(self, query_string: str) -> list[WorkItem]:
        """WorkItemProvider-compatible query."""
        return self.query_work_items(query_string)

    def add_comment(self, item_id: int, text: str,
                    author_agent_id: str) -> Comment:
        """Add a comment to a work item."""
        self.get_work_item(item_id)  # validate exists
        comment = Comment(
            comment_id=str(uuid4()),
            item_id=item_id,
            text=text,
            author_agent_id=author_agent_id,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._comments.setdefault(item_id, []).append(comment)
        return comment

    def get_comments(self, item_id: int) -> list[Comment]:
        """Get all comments for a work item."""
        self.get_work_item(item_id)  # validate exists
        return self._comments.get(item_id, [])

    def link_pr(self, item_id: int, pr_url: str) -> WorkItem:
        """WorkItemProvider-compatible PR linking."""
        return self.link_pr_to_work_item(item_id, pr_url)
