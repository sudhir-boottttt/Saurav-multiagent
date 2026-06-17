"""Property tests and unit tests for Context Manager.

Property tests cover Properties 29-31 from the design document.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.context_manager import ContextManager
from src.knowledge.qdrant_service import QdrantService
from tests.generators import uuid_st, non_empty_str, embedding_st


# ──────────────────────────────────────────────
# Property 29: Context Persistence Round-Trip
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 29: Context Persistence Round-Trip


@given(
    agent_id=uuid_st,
    session_id=uuid_st,
    content=non_empty_str,
    embedding=embedding_st,
)
@settings(max_examples=100)
def test_property_29_context_persistence_round_trip(agent_id, session_id, content, embedding):
    """Persisted task outcomes SHALL be retrievable via load_context."""
    qdrant = QdrantService(vector_size=len(embedding))
    mgr = ContextManager(qdrant)

    mgr.persist_task_outcome(agent_id, session_id, content, embedding, tags=["task_outcome"])
    loaded = mgr.load_context(agent_id, limit=50)

    assert len(loaded) >= 1
    assert any(content in e["content"] for e in loaded)


# ──────────────────────────────────────────────
# Property 30: Conversation Memory Pruning
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 30: Conversation Memory Pruning


def test_property_30_pruning_preserves_tagged():
    """Tagged entries SHALL be retained regardless of age. Untagged old entries pruned."""
    from datetime import UTC, datetime, timedelta

    qdrant = QdrantService(vector_size=3)
    mgr = ContextManager(qdrant)

    # Store old untagged entry (manually set timestamp)
    pid1 = qdrant.store_conversation("agent-1", "s1", "old untagged", [0.1, 0.2, 0.3])
    qdrant._conversation_memory[pid1]["metadata"]["timestamp"] = (
        datetime.now(UTC) - timedelta(days=100)
    ).isoformat()

    # Store old tagged entry
    pid2 = qdrant.store_conversation("agent-1", "s1", "old tagged", [0.1, 0.2, 0.3],
                                      topic_tags=["significant_decision"])
    qdrant._conversation_memory[pid2]["metadata"]["timestamp"] = (
        datetime.now(UTC) - timedelta(days=100)
    ).isoformat()

    # Store recent entry
    qdrant.store_conversation("agent-1", "s1", "recent", [0.1, 0.2, 0.3])

    pruned = mgr.prune_memory(max_age_days=90)
    assert pruned == 1  # Only old untagged removed

    # Tagged entry still exists
    remaining = qdrant.get_recent_context("agent-1", limit=50)
    contents = [t.content for t in remaining]
    assert "old tagged" in contents
    assert "recent" in contents
    assert "old untagged" not in contents


# ──────────────────────────────────────────────
# Property 31: Context Window Summarization
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 31: Context Window Summarization


def test_property_31_summarization_fits_token_limit():
    """Summarized context SHALL fit within the token limit."""
    qdrant = QdrantService(vector_size=3)
    mgr = ContextManager(qdrant, token_limit=100)

    # Create entries that exceed token limit
    entries = [{"content": "x" * 200, "tags": []} for _ in range(5)]

    result = mgr.summarize_context(entries)

    # Result should fit within token limit
    total_tokens = sum(ContextManager._estimate_tokens(e.get("content", "")) for e in result)
    assert total_tokens <= 100

    # Should have at least one entry
    assert len(result) >= 1


def test_property_31_small_context_unchanged():
    """Context within token limit SHALL be returned unchanged."""
    qdrant = QdrantService(vector_size=3)
    mgr = ContextManager(qdrant, token_limit=10000)

    entries = [{"content": "short", "tags": []}]
    result = mgr.summarize_context(entries)
    assert result == entries


class TestContextManager:

    def test_load_empty_context(self):
        qdrant = QdrantService(vector_size=3)
        mgr = ContextManager(qdrant)
        loaded = mgr.load_context("nonexistent-agent")
        assert loaded == []

    def test_checkpoint_save_restore(self):
        qdrant = QdrantService(vector_size=3)
        mgr = ContextManager(qdrant)
        ctx = {"last_task": "bug_fix", "decisions": ["use retry"]}
        mgr.save_checkpoint("agent-1", ctx)
        restored = mgr.restore_checkpoint("agent-1")
        assert restored == ctx

    def test_checkpoint_nonexistent_returns_none(self):
        qdrant = QdrantService(vector_size=3)
        mgr = ContextManager(qdrant)
        assert mgr.restore_checkpoint("nonexistent") is None

    def test_persist_with_tags(self):
        qdrant = QdrantService(vector_size=3)
        mgr = ContextManager(qdrant)
        pid = mgr.persist_task_outcome("a1", "s1", "decision made", [0.1, 0.2, 0.3],
                                        tags=["significant_decision"])
        assert pid is not None
        turns = qdrant.get_recent_context("a1")
        assert any("significant_decision" in t.topic_tags for t in turns)
