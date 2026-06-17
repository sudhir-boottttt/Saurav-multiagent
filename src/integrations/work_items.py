"""Pluggable work item tracking interface.

Provides WorkItemProvider protocol and LocalWorkItemProvider for offline use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    """A work item (feature, bug, task, etc.)."""

    item_id: int
    type: str
    title: str
    description: str
    state: str = "New"
    priority: int = 2
    parent_id: Optional[int] = None
    pr_url: Optional[str] = None
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class Comment:
    """A comment on a work item."""

    comment_id: str
    item_id: int
    text: str
    author_agent_id: str
    timestamp: str


@runtime_checkable
class WorkItemProvider(Protocol):
    """Pluggable work item tracking interface."""

    def create_item(self, type: str, title: str, description: str,
                    fields: dict | None = None) -> WorkItem: ...

    def create_child_item(self, parent_id: int, type: str, title: str,
                          fields: dict | None = None) -> WorkItem: ...

    def update_item(self, item_id: int, fields: dict) -> WorkItem: ...

    def query_items(self, query_string: str) -> list[WorkItem]: ...

    def add_comment(self, item_id: int, text: str,
                    author_agent_id: str) -> Comment: ...

    def get_comments(self, item_id: int) -> list[Comment]: ...

    def link_pr(self, item_id: int, pr_url: str) -> WorkItem: ...


class LocalWorkItemProvider:
    """Stores work items as JSON files for offline use."""

    def __init__(self, base_path: str = "/tmp/agent-repos/") -> None:
        self._dir = Path(base_path) / ".work-items"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._next_id = 1
        self._items: dict[int, WorkItem] = {}
        self._comments: dict[int, list[Comment]] = {}

    def create_item(self, type: str, title: str, description: str,
                    fields: dict | None = None) -> WorkItem:
        item_id = self._next_id
        self._next_id += 1
        wi = WorkItem(item_id=item_id, type=type, title=title,
                      description=description, fields=fields or {})
        self._items[item_id] = wi
        self._save_item(wi)
        return wi

    def create_child_item(self, parent_id: int, type: str, title: str,
                          fields: dict | None = None) -> WorkItem:
        if parent_id not in self._items:
            raise KeyError(f"Parent #{parent_id} not found")
        wi = self.create_item(type, title, f"Child of #{parent_id}", fields)
        wi.parent_id = parent_id
        self._save_item(wi)
        return wi

    def update_item(self, item_id: int, fields: dict) -> WorkItem:
        wi = self._get_item(item_id)
        for k, v in fields.items():
            if k == "state":
                wi.state = v
            elif k == "priority":
                wi.priority = v
            else:
                wi.fields[k] = v
        self._save_item(wi)
        return wi

    def query_items(self, query_string: str) -> list[WorkItem]:
        """Simple query — returns items matching type or title substring."""
        q = query_string.lower()
        return [wi for wi in self._items.values()
                if q in wi.title.lower() or q in wi.type.lower() or str(wi.item_id) == q]

    def add_comment(self, item_id: int, text: str,
                    author_agent_id: str) -> Comment:
        self._get_item(item_id)  # validate exists
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
        self._get_item(item_id)
        return self._comments.get(item_id, [])

    def link_pr(self, item_id: int, pr_url: str) -> WorkItem:
        wi = self._get_item(item_id)
        wi.pr_url = pr_url
        wi.state = "In Review"
        self._save_item(wi)
        return wi

    def _get_item(self, item_id: int) -> WorkItem:
        if item_id not in self._items:
            raise KeyError(f"Work item #{item_id} not found")
        return self._items[item_id]

    def _save_item(self, wi: WorkItem) -> None:
        path = self._dir / f"{wi.item_id}.json"
        data = {
            "item_id": wi.item_id, "type": wi.type, "title": wi.title,
            "description": wi.description, "state": wi.state,
            "priority": wi.priority, "parent_id": wi.parent_id,
            "pr_url": wi.pr_url, "fields": wi.fields,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
