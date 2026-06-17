"""Discussion Thread system for structured multi-agent conversations.

Supports thread lifecycle management, moderator assignment, semantic
similarity search against Knowledge_Base, and synthesis when all
invited agents have contributed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from src.core.models import CodeReference, Contribution, DiscussionThread
from src.knowledge.qdrant_service import QdrantService

if TYPE_CHECKING:
    from src.infrastructure.storage import StorageBackend

logger = logging.getLogger(__name__)


class DiscussionThreadError(Exception):
    """Base error for discussion thread operations."""


class ThreadNotFoundError(DiscussionThreadError):
    """Raised when a thread_id doesn't exist."""


class ThreadClosedError(DiscussionThreadError):
    """Raised when trying to modify a closed thread."""


class DuplicateContributionError(DiscussionThreadError):
    """Raised when an uninvited agent tries to contribute."""


class DiscussionThreadManager:
    """Manages discussion thread lifecycle with Knowledge_Base integration.

    Threads map to AutoGen GroupChat sessions. Moderator assignment follows:
    - Cross-domain threads: PM Agent moderates
    - Single-domain threads: Tech Lead moderates
    """

    SIMILARITY_THRESHOLD = 0.85

    def __init__(
        self,
        qdrant_service: Optional[QdrantService] = None,
        pm_agent_id: Optional[str] = None,
        domain_tech_leads: Optional[dict[str, str]] = None,
        storage: Optional[StorageBackend] = None,
    ) -> None:
        """
        Args:
            qdrant_service: Optional Qdrant service for similarity search and storage.
            pm_agent_id: The PM agent's ID (used as moderator for cross-domain threads).
            domain_tech_leads: Mapping of domain name -> Tech Lead agent_id.
            storage: Optional StorageBackend for persisting threads.
        """
        self._threads: dict[str, DiscussionThread] = {}
        self._qdrant = qdrant_service
        self._pm_agent_id = pm_agent_id or "pm-agent"
        self._domain_tech_leads = domain_tech_leads or {}
        self._storage = storage
        # Track which agents belong to which domains for moderator assignment
        self._agent_domains: dict[str, str] = {}

    def register_agent_domain(self, agent_id: str, domain: str) -> None:
        """Register an agent's domain for moderator assignment logic."""
        self._agent_domains[agent_id] = domain

    def create_thread(
        self,
        topic: str,
        creator_id: str,
        invited_agents: list[str],
        is_cross_domain: bool = False,
        moderator_id: Optional[str] = None,
    ) -> DiscussionThread:
        """Create a new discussion thread.

        Moderator assignment:
        - If moderator_id is provided, use it directly.
        - If is_cross_domain is True, PM Agent moderates.
        - Otherwise, the creator's domain Tech Lead moderates.
        """
        if not topic or not topic.strip():
            raise DiscussionThreadError("Thread topic must be non-empty")
        if not creator_id or not creator_id.strip():
            raise DiscussionThreadError("Creator ID must be non-empty")

        # Determine moderator
        if moderator_id:
            resolved_moderator = moderator_id
        elif is_cross_domain:
            resolved_moderator = self._pm_agent_id
        else:
            # Single-domain: use creator's domain Tech Lead, fallback to creator
            creator_domain = self._agent_domains.get(creator_id)
            resolved_moderator = self._domain_tech_leads.get(creator_domain, creator_id) if creator_domain else creator_id

        thread = DiscussionThread(
            thread_id=str(uuid4()),
            topic=topic,
            creator_id=creator_id,
            invited_agents=invited_agents,
            contributions=[],
            is_cross_domain=is_cross_domain,
            moderator_id=resolved_moderator,
            summary=None,
            status="open",
            created_at=datetime.now(UTC),
            closed_at=None,
        )
        self._threads[thread.thread_id] = thread
        if self._storage:
            self._storage.save("threads", thread.thread_id, thread.model_dump(mode="json"))
        logger.info("Created thread %s: %s (moderator=%s)", thread.thread_id, topic, resolved_moderator)
        return thread

    def add_contribution(
        self,
        thread_id: str,
        agent_id: str,
        content: str,
        code_refs: Optional[list[dict[str, Any]]] = None,
    ) -> Contribution:
        """Add a contribution to a thread.

        Raises ThreadNotFoundError if thread doesn't exist.
        Raises ThreadClosedError if thread is closed.
        """
        thread = self._get_thread_or_raise(thread_id)
        if thread.status == "closed":
            raise ThreadClosedError(f"Thread {thread_id} is closed")

        if not content or not content.strip():
            raise DiscussionThreadError("Contribution content must be non-empty")

        # Build code references
        refs = []
        for ref_data in (code_refs or []):
            refs.append(CodeReference(
                file_path=ref_data["file_path"],
                line_start=ref_data["line_start"],
                line_end=ref_data["line_end"],
            ))

        contribution = Contribution(
            agent_id=agent_id,
            timestamp=datetime.now(UTC),
            content=content,
            code_refs=refs,
        )
        thread.contributions.append(contribution)
        logger.info("Agent %s contributed to thread %s", agent_id, thread_id)
        return contribution

    def get_thread(self, thread_id: str) -> DiscussionThread:
        """Get a thread by ID. Raises ThreadNotFoundError if not found."""
        return self._get_thread_or_raise(thread_id)

    def all_invited_contributed(self, thread_id: str) -> bool:
        """Check if all invited agents have contributed at least once."""
        thread = self._get_thread_or_raise(thread_id)
        contributed_agents = {c.agent_id for c in thread.contributions}
        return all(agent in contributed_agents for agent in thread.invited_agents)

    def synthesize(self, thread_id: str) -> str:
        """Synthesize a thread summary when all invited agents have contributed.

        Returns the synthesis summary string.
        Raises DiscussionThreadError if not all agents have contributed.
        """
        thread = self._get_thread_or_raise(thread_id)

        if not self.all_invited_contributed(thread_id):
            raise DiscussionThreadError(
                f"Cannot synthesize thread {thread_id}: not all invited agents have contributed"
            )

        # Build synthesis from contributions
        parts = []
        for c in sorted(thread.contributions, key=lambda x: x.timestamp):
            refs_str = ""
            if c.code_refs:
                refs_str = " [refs: " + ", ".join(
                    f"{r.file_path}:{r.line_start}-{r.line_end}" for r in c.code_refs
                ) + "]"
            parts.append(f"- {c.agent_id}: {c.content}{refs_str}")

        summary = f"Thread: {thread.topic}\nModerator: {thread.moderator_id}\n\nContributions:\n" + "\n".join(parts)
        thread.summary = summary
        thread.status = "synthesized"
        logger.info("Synthesized thread %s", thread_id)
        return summary

    def find_similar(
        self,
        topic_embedding: list[float],
        threshold: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Find past discussion threads with similarity above threshold.

        Queries the Knowledge_Base collection in Qdrant.
        Returns list of dicts with thread metadata and similarity score.
        """
        if not self._qdrant:
            return []

        results = self._qdrant.search_knowledge(topic_embedding, top_k=10)
        similar = []
        for r in results:
            if r.score >= threshold and r.metadata.get("type") == "discussion_summary":
                similar.append({
                    "point_id": r.point_id,
                    "thread_id": r.metadata.get("thread_id"),
                    "topic": r.metadata.get("topic"),
                    "summary": r.metadata.get("summary"),
                    "score": r.score,
                })
        return similar

    def close_thread(self, thread_id: str, summary: Optional[str] = None) -> DiscussionThread:
        """Close a thread and optionally store its summary in Knowledge_Base.

        If summary is provided, it overrides any existing synthesis summary.
        """
        thread = self._get_thread_or_raise(thread_id)
        if summary:
            thread.summary = summary
        thread.status = "closed"
        thread.closed_at = datetime.now(UTC)
        logger.info("Closed thread %s", thread_id)
        return thread

    def store_thread_summary(
        self,
        thread_id: str,
        embedding: list[float],
    ) -> Optional[str]:
        """Store a completed thread's summary as an embedding in Knowledge_Base.

        Returns the Qdrant point_id, or None if no Qdrant service or no summary.
        """
        thread = self._get_thread_or_raise(thread_id)
        if not self._qdrant or not thread.summary:
            return None

        metadata = {
            "type": "discussion_summary",
            "thread_id": thread.thread_id,
            "topic": thread.topic,
            "summary": thread.summary,
            "creator_id": thread.creator_id,
            "is_cross_domain": thread.is_cross_domain,
            "moderator_id": thread.moderator_id,
            "status": thread.status,
            "created_at": thread.created_at.isoformat(),
            "closed_at": thread.closed_at.isoformat() if thread.closed_at else None,
            "contributor_count": len({c.agent_id for c in thread.contributions}),
        }
        point_id = self._qdrant.store_knowledge(embedding, metadata)
        logger.info("Stored thread %s summary in Knowledge_Base as %s", thread_id, point_id)
        return point_id

    def _get_thread_or_raise(self, thread_id: str) -> DiscussionThread:
        thread = self._threads.get(thread_id)
        if not thread:
            raise ThreadNotFoundError(f"Thread {thread_id} not found")
        return thread
