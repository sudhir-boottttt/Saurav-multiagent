"""Agent context persistence and session management.

Handles loading context from Conversation_Memory, persisting task
outcomes, checkpoint-based recovery, memory pruning, and context
window summarization.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Optional

from src.knowledge.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LIMIT = 50
DEFAULT_MAX_AGE_DAYS = 90
DEFAULT_TOKEN_LIMIT = 8000
PRESERVE_TAGS = ["significant_decision", "architectural_context"]


class ContextManager:
    """Manages agent context loading, persistence, and summarization."""

    def __init__(
        self,
        qdrant_service: QdrantService,
        token_limit: int = DEFAULT_TOKEN_LIMIT,
    ) -> None:
        self._qdrant = qdrant_service
        self._token_limit = token_limit
        self._checkpoints: dict[str, dict[str, Any]] = {}

    def load_context(self, agent_id: str, limit: int = DEFAULT_CONTEXT_LIMIT) -> list[dict[str, Any]]:
        """Load last N conversation turns for an agent from Conversation_Memory.

        Returns list of dicts with content and tags.
        """
        turns = self._qdrant.get_recent_context(agent_id, limit=limit)
        return [{"content": t.content, "tags": t.topic_tags, "timestamp": t.timestamp.isoformat()} for t in turns]

    def persist_task_outcome(
        self,
        agent_id: str,
        session_id: str,
        content: str,
        embedding: list[float],
        tags: list[str] | None = None,
    ) -> str:
        """Persist a task outcome to Conversation_Memory. Returns point_id."""
        return self._qdrant.store_conversation(
            agent_id=agent_id,
            session_id=session_id,
            content=content,
            embedding=embedding,
            topic_tags=tags or [],
        )

    def save_checkpoint(self, agent_id: str, context: dict[str, Any]) -> None:
        """Save a checkpoint for recovery on agent restart."""
        self._checkpoints[agent_id] = {
            "context": context,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        logger.info("Saved checkpoint for agent %s", agent_id)

    def restore_checkpoint(self, agent_id: str) -> Optional[dict[str, Any]]:
        """Restore context from last checkpoint. Returns None if no checkpoint."""
        cp = self._checkpoints.get(agent_id)
        if cp:
            logger.info("Restored checkpoint for agent %s", agent_id)
            return cp["context"]
        return None

    def prune_memory(self, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> int:
        """Prune old conversation memory entries. Returns count pruned."""
        return self._qdrant.prune_old_entries(
            max_age_days=max_age_days,
            preserve_tags=PRESERVE_TAGS,
        )

    def summarize_context(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Summarize older context entries when approaching token limit.

        Keeps recent entries in full, summarizes older ones.
        Returns the summarized context list.
        """
        total_tokens = sum(self._estimate_tokens(e.get("content", "")) for e in entries)

        if total_tokens <= self._token_limit:
            return entries

        # Keep recent entries in full, summarize older ones
        result = []
        running_tokens = 0

        # Process from newest to oldest
        reversed_entries = list(reversed(entries))
        full_entries = []
        to_summarize = []

        for entry in reversed_entries:
            tokens = self._estimate_tokens(entry.get("content", ""))
            if running_tokens + tokens <= self._token_limit * 0.7:
                full_entries.append(entry)
                running_tokens += tokens
            else:
                to_summarize.append(entry)

        # Summarize older entries
        if to_summarize:
            summary_content = "Summary of older context: " + "; ".join(
                e.get("content", "")[:100] for e in to_summarize
            )
            summary_entry = {
                "content": summary_content[:self._token_limit // 4],
                "tags": ["summarized"],
                "is_summary": True,
            }
            result.append(summary_entry)

        # Add full entries in original order
        result.extend(reversed(full_entries))
        return result

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)
