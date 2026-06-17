"""Property-based and unit tests for the Qdrant Knowledge Layer."""

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.knowledge.qdrant_service import QdrantService


# --- Property Tests ---
# Feature: multi-agent-platform, Property 17: Code Index Round-Trip


def test_code_index_round_trip():
    """Indexed code chunks should be retrievable via semantic search."""
    svc = QdrantService(vector_size=3)
    embedding = [0.5, 0.3, 0.1]
    svc.index_code("receiving-svc", [{
        "chunk_id": "c1", "file_path": "src/main.py", "language": "python",
        "commit_hash": "abc1234", "chunk_type": "function", "content": "def hello(): pass",
        "start_line": 1, "end_line": 2, "embedding": embedding,
    }])
    results = svc.search_code(embedding, top_k=5)
    assert len(results) >= 1
    assert results[0].file_path == "src/main.py"
    assert results[0].repo == "receiving-svc"
    assert results[0].language == "python"
    assert results[0].commit_hash == "abc1234"


# Feature: multi-agent-platform, Property 18: Conversation Memory Metadata


def test_conversation_memory_metadata_completeness():
    """Stored conversation turns must have agent_id, session_id, timestamp, topic_tags."""
    svc = QdrantService()
    svc.store_conversation("agent-1", "session-1", "hello world", [0.1, 0.2], ["debug"])
    turns = svc.get_recent_context("agent-1")
    assert len(turns) == 1
    t = turns[0]
    assert t.agent_id == "agent-1"
    assert t.session_id == "session-1"
    assert t.timestamp is not None
    assert t.topic_tags == ["debug"]


# Feature: multi-agent-platform, Property 19: Context Assembly


def test_context_assembly_from_all_collections():
    """Context assembly should pull from all three collections."""
    svc = QdrantService(vector_size=3)
    emb = [0.5, 0.3, 0.1]

    # Add code
    svc.index_code("repo1", [{"chunk_id": "c1", "file_path": "f.py", "language": "python",
                               "commit_hash": "abc", "chunk_type": "function",
                               "content": "code here", "embedding": emb}])
    # Add knowledge
    svc.store_knowledge(emb, {"topic": "architecture decision"})
    # Add conversation
    svc.store_conversation("agent-1", "s1", "discussed bug", emb, ["bug"])

    ctx = svc.assemble_context(emb, "agent-1")
    assert len(ctx["code_context"]) >= 1
    assert len(ctx["knowledge_context"]) >= 1
    assert len(ctx["conversation_context"]) >= 1


# Feature: multi-agent-platform, Property 30: Conversation Memory Pruning


def test_pruning_removes_old_untagged_entries():
    """Entries older than 90 days without preserve tags should be pruned."""
    svc = QdrantService()
    # Store an old entry
    pid = svc.store_conversation("agent-1", "s1", "old msg", [0.1], ["general"])
    # Manually age it
    svc._conversation_memory[pid]["metadata"]["timestamp"] = (
        datetime.now(UTC) - timedelta(days=91)
    ).isoformat()

    # Store a recent entry
    svc.store_conversation("agent-1", "s2", "new msg", [0.2], ["general"])

    pruned = svc.prune_old_entries(max_age_days=90)
    assert pruned == 1
    assert len(svc.get_recent_context("agent-1")) == 1


def test_pruning_preserves_tagged_entries():
    """Entries tagged as significant_decision should survive pruning."""
    svc = QdrantService()
    pid = svc.store_conversation("agent-1", "s1", "important", [0.1], ["significant_decision"])
    svc._conversation_memory[pid]["metadata"]["timestamp"] = (
        datetime.now(UTC) - timedelta(days=120)
    ).isoformat()

    pruned = svc.prune_old_entries(max_age_days=90)
    assert pruned == 0
    assert len(svc.get_recent_context("agent-1")) == 1


# --- Unit Tests ---


def test_search_code_with_repo_filter():
    svc = QdrantService(vector_size=3)
    emb = [0.5, 0.3, 0.1]
    svc.index_code("repo-a", [{"chunk_id": "c1", "embedding": emb, "file_path": "a.py",
                                "language": "python", "commit_hash": "abc", "content": "a"}])
    svc.index_code("repo-b", [{"chunk_id": "c2", "embedding": emb, "file_path": "b.py",
                                "language": "python", "commit_hash": "def", "content": "b"}])
    results = svc.search_code(emb, repo_filter="repo-a")
    assert len(results) == 1
    assert results[0].repo == "repo-a"


def test_knowledge_base_search():
    svc = QdrantService(vector_size=3)
    emb = [0.5, 0.3, 0.1]
    svc.store_knowledge(emb, {"topic": "test discussion"})
    results = svc.search_knowledge(emb, top_k=5)
    assert len(results) == 1
    assert results[0].metadata["topic"] == "test discussion"
    assert results[0].score > 0.99  # same vector should be ~1.0


def test_recent_context_limit():
    svc = QdrantService()
    for i in range(10):
        svc.store_conversation("agent-1", "s1", f"msg {i}", [0.1], [])
    turns = svc.get_recent_context("agent-1", limit=5)
    assert len(turns) == 5


def test_cosine_similarity_identical_vectors():
    assert abs(QdrantService._cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 0.001


def test_cosine_similarity_orthogonal_vectors():
    assert abs(QdrantService._cosine_similarity([1, 0], [0, 1])) < 0.001


def test_cosine_similarity_empty_vectors():
    assert QdrantService._cosine_similarity([], []) == 0.0
