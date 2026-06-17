"""Property tests and unit tests for the Discussion Thread system.

Property tests cover Properties 9-12 from the design document.
Unit tests cover thread lifecycle edge cases.
"""

import math
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.core.models import DiscussionThread
from src.infrastructure.discussion_thread import (
    DiscussionThreadError,
    DiscussionThreadManager,
    ThreadClosedError,
    ThreadNotFoundError,
)
from src.knowledge.qdrant_service import QdrantService
from tests.generators import uuid_st, non_empty_str, contribution_st


# ──────────────────────────────────────────────
# Property 9: Discussion Thread Contribution Ordering
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 9: Discussion Thread Contribution Ordering


@given(
    topic=non_empty_str,
    creator_id=uuid_st,
    agent_ids=st.lists(uuid_st, min_size=2, max_size=5, unique=True),
    contents=st.lists(non_empty_str, min_size=2, max_size=5),
)
@settings(max_examples=100)
def test_property_9_contribution_ordering(topic, creator_id, agent_ids, contents):
    """Contributions SHALL be ordered by timestamp, each containing
    agent_id, timestamp, content, and code_refs. No contribution
    timestamp SHALL be earlier than the thread creation timestamp.
    """
    mgr = DiscussionThreadManager()
    thread = mgr.create_thread(topic=topic, creator_id=creator_id, invited_agents=agent_ids)
    creation_time = thread.created_at

    # Add contributions from different agents
    n = min(len(agent_ids), len(contents))
    for i in range(n):
        mgr.add_contribution(thread.thread_id, agent_ids[i], contents[i])

    result = mgr.get_thread(thread.thread_id)

    # Verify ordering and field completeness
    for i, c in enumerate(result.contributions):
        assert c.agent_id, "agent_id must be non-empty"
        assert c.timestamp is not None, "timestamp must be present"
        assert c.content, "content must be non-empty"
        assert isinstance(c.code_refs, list), "code_refs must be a list"
        # No contribution before thread creation
        assert c.timestamp >= creation_time, (
            f"Contribution timestamp {c.timestamp} is before thread creation {creation_time}"
        )

    # Verify ordering by timestamp
    timestamps = [c.timestamp for c in result.contributions]
    assert timestamps == sorted(timestamps), "Contributions must be ordered by timestamp"


# ──────────────────────────────────────────────
# Property 10: Discussion Thread Synthesis Trigger
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 10: Discussion Thread Synthesis Trigger


@given(
    topic=non_empty_str,
    creator_id=uuid_st,
    agent_ids=st.lists(uuid_st, min_size=2, max_size=4, unique=True),
)
@settings(max_examples=100)
def test_property_10_synthesis_trigger(topic, creator_id, agent_ids):
    """Synthesis SHALL not be produced before all invited agents have contributed.
    When all have contributed, synthesis SHALL produce a summary.
    """
    mgr = DiscussionThreadManager()
    thread = mgr.create_thread(topic=topic, creator_id=creator_id, invited_agents=agent_ids)

    # Before all contribute, synthesis should fail
    with pytest.raises(DiscussionThreadError, match="not all invited agents"):
        mgr.synthesize(thread.thread_id)

    # Add contributions from all but last agent
    for aid in agent_ids[:-1]:
        mgr.add_contribution(thread.thread_id, aid, f"Analysis from {aid}")

    # Still should fail
    with pytest.raises(DiscussionThreadError, match="not all invited agents"):
        mgr.synthesize(thread.thread_id)

    # Last agent contributes
    mgr.add_contribution(thread.thread_id, agent_ids[-1], f"Analysis from {agent_ids[-1]}")

    # Now synthesis should succeed
    summary = mgr.synthesize(thread.thread_id)
    assert summary, "Synthesis must produce a non-empty summary"
    assert mgr.get_thread(thread.thread_id).status == "synthesized"
    assert mgr.get_thread(thread.thread_id).summary == summary


# ──────────────────────────────────────────────
# Property 11: Cross-Domain Discussion Moderation
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 11: Cross-Domain Discussion Moderation


@given(
    topic=non_empty_str,
    creator_id=uuid_st,
    invited=st.lists(uuid_st, min_size=1, max_size=3),
)
@settings(max_examples=100)
def test_property_11_cross_domain_moderation(topic, creator_id, invited):
    """Cross-domain threads SHALL have PM Agent as moderator.
    Single-domain threads SHALL have the domain's Tech Lead as moderator.
    """
    pm_id = "pm-agent-001"
    tl_inbound = "tl-inbound-001"
    tl_outbound = "tl-outbound-001"

    mgr = DiscussionThreadManager(
        pm_agent_id=pm_id,
        domain_tech_leads={"inbound": tl_inbound, "outbound": tl_outbound},
    )
    mgr.register_agent_domain(creator_id, "inbound")

    # Cross-domain thread -> PM moderates
    cross_thread = mgr.create_thread(
        topic=topic, creator_id=creator_id, invited_agents=invited, is_cross_domain=True,
    )
    assert cross_thread.moderator_id == pm_id, "Cross-domain thread must have PM as moderator"

    # Single-domain thread -> Tech Lead moderates
    single_thread = mgr.create_thread(
        topic=topic, creator_id=creator_id, invited_agents=invited, is_cross_domain=False,
    )
    assert single_thread.moderator_id == tl_inbound, "Single-domain thread must have Tech Lead as moderator"


# ──────────────────────────────────────────────
# Property 12: Discussion Thread Knowledge Base Round-Trip
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 12: Discussion Thread Knowledge Base Round-Trip


@given(
    topic=non_empty_str,
    creator_id=uuid_st,
    agent_ids=st.lists(uuid_st, min_size=1, max_size=3, unique=True),
)
@settings(max_examples=100)
def test_property_12_knowledge_base_round_trip(topic, creator_id, agent_ids):
    """Storing a thread summary embedding in Knowledge_Base and querying
    with the same embedding SHALL return a result with cosine > 0.85.
    """
    qdrant = QdrantService(vector_size=5)
    mgr = DiscussionThreadManager(qdrant_service=qdrant)

    thread = mgr.create_thread(topic=topic, creator_id=creator_id, invited_agents=agent_ids)

    # All agents contribute
    for aid in agent_ids:
        mgr.add_contribution(thread.thread_id, aid, f"Input from {aid}")

    # Synthesize and close
    mgr.synthesize(thread.thread_id)
    mgr.close_thread(thread.thread_id)

    # Use a normalized embedding vector
    embedding = [0.5, 0.3, 0.8, 0.1, 0.6]
    norm = math.sqrt(sum(x * x for x in embedding))
    embedding = [x / norm for x in embedding]

    # Store in Knowledge_Base
    point_id = mgr.store_thread_summary(thread.thread_id, embedding)
    assert point_id is not None, "Should return a point_id"

    # Query with same embedding -> should get cosine ~1.0
    results = mgr.find_similar(embedding, threshold=0.85)
    assert len(results) >= 1, "Should find the stored thread"
    assert results[0]["thread_id"] == thread.thread_id
    assert results[0]["score"] >= 0.85


# ──────────────────────────────────────────────
# Unit Tests: Thread Lifecycle Edge Cases
# ──────────────────────────────────────────────


class TestDiscussionThreadLifecycle:
    """Unit tests for thread lifecycle edge cases."""

    def test_create_thread_basic(self):
        mgr = DiscussionThreadManager()
        thread = mgr.create_thread("Bug in receiving-svc", "tl-001", ["dev-001", "dev-002"])
        assert thread.status == "open"
        assert thread.topic == "Bug in receiving-svc"
        assert thread.creator_id == "tl-001"
        assert len(thread.invited_agents) == 2
        assert thread.summary is None

    def test_create_thread_empty_topic_raises(self):
        mgr = DiscussionThreadManager()
        with pytest.raises(DiscussionThreadError, match="topic must be non-empty"):
            mgr.create_thread("", "tl-001", ["dev-001"])

    def test_create_thread_empty_creator_raises(self):
        mgr = DiscussionThreadManager()
        with pytest.raises(DiscussionThreadError, match="Creator ID must be non-empty"):
            mgr.create_thread("topic", "", ["dev-001"])

    def test_get_nonexistent_thread_raises(self):
        mgr = DiscussionThreadManager()
        with pytest.raises(ThreadNotFoundError):
            mgr.get_thread("nonexistent-id")

    def test_contribute_to_closed_thread_raises(self):
        mgr = DiscussionThreadManager()
        thread = mgr.create_thread("topic", "tl-001", ["dev-001"])
        mgr.add_contribution(thread.thread_id, "dev-001", "my input")
        mgr.close_thread(thread.thread_id, "done")
        with pytest.raises(ThreadClosedError):
            mgr.add_contribution(thread.thread_id, "dev-001", "more input")

    def test_contribute_empty_content_raises(self):
        mgr = DiscussionThreadManager()
        thread = mgr.create_thread("topic", "tl-001", ["dev-001"])
        with pytest.raises(DiscussionThreadError, match="content must be non-empty"):
            mgr.add_contribution(thread.thread_id, "dev-001", "")

    def test_contribute_with_code_refs(self):
        mgr = DiscussionThreadManager()
        thread = mgr.create_thread("topic", "tl-001", ["dev-001"])
        refs = [{"file_path": "src/main.py", "line_start": 10, "line_end": 20}]
        c = mgr.add_contribution(thread.thread_id, "dev-001", "found issue here", code_refs=refs)
        assert len(c.code_refs) == 1
        assert c.code_refs[0].file_path == "src/main.py"

    def test_all_invited_contributed_check(self):
        mgr = DiscussionThreadManager()
        thread = mgr.create_thread("topic", "tl-001", ["dev-001", "dev-002"])
        assert not mgr.all_invited_contributed(thread.thread_id)
        mgr.add_contribution(thread.thread_id, "dev-001", "input 1")
        assert not mgr.all_invited_contributed(thread.thread_id)
        mgr.add_contribution(thread.thread_id, "dev-002", "input 2")
        assert mgr.all_invited_contributed(thread.thread_id)

    def test_synthesize_includes_all_contributions(self):
        mgr = DiscussionThreadManager()
        thread = mgr.create_thread("API redesign", "tl-001", ["dev-001", "dev-002"])
        mgr.add_contribution(thread.thread_id, "dev-001", "Need to update endpoints")
        mgr.add_contribution(thread.thread_id, "dev-002", "Backward compat concern")
        summary = mgr.synthesize(thread.thread_id)
        assert "dev-001" in summary
        assert "dev-002" in summary
        assert "Need to update endpoints" in summary
        assert "Backward compat concern" in summary

    def test_close_thread_sets_status_and_timestamp(self):
        mgr = DiscussionThreadManager()
        thread = mgr.create_thread("topic", "tl-001", ["dev-001"])
        closed = mgr.close_thread(thread.thread_id, "Final summary")
        assert closed.status == "closed"
        assert closed.closed_at is not None
        assert closed.summary == "Final summary"

    def test_store_summary_without_qdrant_returns_none(self):
        mgr = DiscussionThreadManager(qdrant_service=None)
        thread = mgr.create_thread("topic", "tl-001", ["dev-001"])
        mgr.add_contribution(thread.thread_id, "dev-001", "input")
        mgr.synthesize(thread.thread_id)
        result = mgr.store_thread_summary(thread.thread_id, [0.1, 0.2])
        assert result is None

    def test_find_similar_without_qdrant_returns_empty(self):
        mgr = DiscussionThreadManager(qdrant_service=None)
        results = mgr.find_similar([0.1, 0.2, 0.3])
        assert results == []

    def test_moderator_explicit_override(self):
        mgr = DiscussionThreadManager(pm_agent_id="pm-001")
        thread = mgr.create_thread(
            "topic", "tl-001", ["dev-001"],
            is_cross_domain=True, moderator_id="custom-mod",
        )
        assert thread.moderator_id == "custom-mod"

    def test_moderator_fallback_to_creator(self):
        """When no domain mapping exists, moderator falls back to creator."""
        mgr = DiscussionThreadManager(pm_agent_id="pm-001")
        thread = mgr.create_thread("topic", "unknown-agent", ["dev-001"], is_cross_domain=False)
        assert thread.moderator_id == "unknown-agent"
