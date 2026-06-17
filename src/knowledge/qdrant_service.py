"""Qdrant service managing three vector collections.

Collections:
- Code_Index: code chunk embeddings per repo
- Knowledge_Base: discussion summaries, architectural decisions
- Conversation_Memory: agent conversation turns
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class CodeResult:
    """A search result from the Code_Index collection."""

    def __init__(self, chunk_id: str, repo: str, file_path: str, language: str,
                 commit_hash: str, content: str, score: float):
        self.chunk_id = chunk_id
        self.repo = repo
        self.file_path = file_path
        self.language = language
        self.commit_hash = commit_hash
        self.content = content
        self.score = score


class KnowledgeResult:
    """A search result from the Knowledge_Base collection."""

    def __init__(self, point_id: str, metadata: dict[str, Any], score: float):
        self.point_id = point_id
        self.metadata = metadata
        self.score = score


class ConversationTurn:
    """A stored conversation turn from Conversation_Memory."""

    def __init__(self, point_id: str, agent_id: str, session_id: str,
                 timestamp: datetime, topic_tags: list[str], content: str):
        self.point_id = point_id
        self.agent_id = agent_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.topic_tags = topic_tags
        self.content = content


class QdrantService:
    """In-memory Qdrant-compatible service for the three collections.

    For production, replace with actual qdrant_client.QdrantClient calls.
    The interface remains identical.
    """

    def __init__(self, vector_size: int = 384) -> None:
        self._vector_size = vector_size
        # In-memory stores: {point_id: {embedding, metadata}}
        self._code_index: dict[str, dict[str, Any]] = {}
        self._knowledge_base: dict[str, dict[str, Any]] = {}
        self._conversation_memory: dict[str, dict[str, Any]] = {}

    # --- Code Index ---

    def index_code(self, repo: str, chunks: list[dict[str, Any]]) -> int:
        """Index code chunks into Code_Index. Returns count indexed."""
        count = 0
        for chunk in chunks:
            point_id = chunk.get("chunk_id", str(uuid4()))
            self._code_index[point_id] = {
                "embedding": chunk.get("embedding", []),
                "metadata": {
                    "repo": repo,
                    "file_path": chunk.get("file_path", ""),
                    "language": chunk.get("language", ""),
                    "commit_hash": chunk.get("commit_hash", ""),
                    "chunk_type": chunk.get("chunk_type", ""),
                    "content": chunk.get("content", ""),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "indexed_at": datetime.now(UTC).isoformat(),
                },
            }
            count += 1
        return count

    def search_code(self, query_embedding: list[float], repo_filter: Optional[str] = None,
                    top_k: int = 10) -> list[CodeResult]:
        """Search Code_Index by semantic similarity. Returns top-k results."""
        scored = []
        for pid, data in self._code_index.items():
            meta = data["metadata"]
            if repo_filter and meta["repo"] != repo_filter:
                continue
            score = self._cosine_similarity(query_embedding, data["embedding"])
            scored.append((score, pid, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            CodeResult(
                chunk_id=pid, repo=meta["repo"], file_path=meta["file_path"],
                language=meta["language"], commit_hash=meta["commit_hash"],
                content=meta["content"], score=score,
            )
            for score, pid, meta in scored[:top_k]
        ]

    # --- Knowledge Base ---

    def store_knowledge(self, embedding: list[float], metadata: dict[str, Any]) -> str:
        """Store an embedding in the Knowledge_Base. Returns point_id."""
        point_id = str(uuid4())
        self._knowledge_base[point_id] = {"embedding": embedding, "metadata": metadata}
        return point_id

    def search_knowledge(self, query_embedding: list[float], top_k: int = 10) -> list[KnowledgeResult]:
        """Search Knowledge_Base by semantic similarity."""
        scored = []
        for pid, data in self._knowledge_base.items():
            score = self._cosine_similarity(query_embedding, data["embedding"])
            scored.append((score, pid, data["metadata"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            KnowledgeResult(point_id=pid, metadata=meta, score=score)
            for score, pid, meta in scored[:top_k]
        ]

    def delete_knowledge_by_source(self, source_file: str) -> int:
        """Remove all Knowledge_Base entries matching a source file. Returns count removed."""
        to_remove = [
            pid for pid, data in self._knowledge_base.items()
            if data["metadata"].get("source_file") == source_file
        ]
        for pid in to_remove:
            del self._knowledge_base[pid]
        return len(to_remove)

    # --- Conversation Memory ---

    def store_conversation(self, agent_id: str, session_id: str, content: str,
                           embedding: list[float], topic_tags: list[str] | None = None) -> str:
        """Store a conversation turn. Returns point_id."""
        point_id = str(uuid4())
        self._conversation_memory[point_id] = {
            "embedding": embedding,
            "metadata": {
                "agent_id": agent_id,
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "topic_tags": topic_tags or [],
                "content": content,
            },
        }
        return point_id

    def get_recent_context(self, agent_id: str, limit: int = 50) -> list[ConversationTurn]:
        """Get recent conversation turns for an agent."""
        turns = []
        for pid, data in self._conversation_memory.items():
            meta = data["metadata"]
            if meta["agent_id"] == agent_id:
                turns.append(ConversationTurn(
                    point_id=pid, agent_id=meta["agent_id"],
                    session_id=meta["session_id"],
                    timestamp=datetime.fromisoformat(meta["timestamp"]),
                    topic_tags=meta["topic_tags"], content=meta["content"],
                ))
        turns.sort(key=lambda t: t.timestamp, reverse=True)
        return turns[:limit]

    def prune_old_entries(self, max_age_days: int = 90,
                          preserve_tags: list[str] | None = None) -> int:
        """Prune conversation memory entries older than max_age_days.

        Entries tagged with any of preserve_tags are retained.
        Returns count of pruned entries.
        """
        preserve_tags = preserve_tags or ["significant_decision", "architectural_context"]
        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
        to_remove = []
        for pid, data in self._conversation_memory.items():
            meta = data["metadata"]
            ts = datetime.fromisoformat(meta["timestamp"])
            tags = set(meta.get("topic_tags", []))
            if ts < cutoff and not tags.intersection(preserve_tags):
                to_remove.append(pid)
        for pid in to_remove:
            del self._conversation_memory[pid]
        return len(to_remove)

    # --- Context Assembly ---

    def assemble_context(self, query_embedding: list[float], agent_id: str,
                         code_top_k: int = 5, knowledge_top_k: int = 5,
                         memory_limit: int = 10) -> dict[str, Any]:
        """Assemble context from all three collections for an agent's task."""
        code_results = self.search_code(query_embedding, top_k=code_top_k)
        knowledge_results = self.search_knowledge(query_embedding, top_k=knowledge_top_k)
        memory_turns = self.get_recent_context(agent_id, limit=memory_limit)
        return {
            "code_context": [{"file": r.file_path, "content": r.content, "repo": r.repo} for r in code_results],
            "knowledge_context": [r.metadata for r in knowledge_results],
            "conversation_context": [{"content": t.content, "tags": t.topic_tags} for t in memory_turns],
        }

    # --- Utility ---

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
