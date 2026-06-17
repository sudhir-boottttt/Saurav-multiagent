"""Context assembler for enriching agent prompts with real data.

Queries Qdrant Code_Index and Knowledge_Base, retrieves conversation memory,
and injects grounding instructions into every agent prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.knowledge.embedding import EmbeddingProvider
from src.knowledge.embedding_cache import EmbeddingCache
from src.knowledge.qdrant_service import CodeResult, KnowledgeResult, QdrantService

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Retrieves and injects relevant context into agent prompts."""

    GROUNDING_INSTRUCTION = (
        "Base your response ONLY on the code snippets, search results, and data "
        "provided in this context. If you do not have sufficient data to answer "
        "a question, explicitly state what information is missing rather than "
        "guessing or making assumptions."
    )

    def __init__(
        self,
        qdrant: Optional[QdrantService] = None,
        embedding: Optional[EmbeddingProvider] = None,
        cache: Optional[EmbeddingCache] = None,
    ) -> None:
        self._qdrant = qdrant
        self._embedding = embedding
        self._cache = cache

    def assemble(
        self,
        task_description: str,
        agent_role: str = "",
        agent_id: str = "",
        assigned_repos: list[str] | None = None,
        system_prompt: str = "",
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Build enriched system prompt with code, knowledge, and memory context."""
        parts = []

        # Start with grounding instruction
        parts.append(self.GROUNDING_INSTRUCTION)
        parts.append("")

        # Original system prompt
        if system_prompt:
            parts.append(system_prompt)
            parts.append("")

        # Code context from Qdrant
        code_context = self._search_code_index(
            task_description,
            repo_filter=assigned_repos,
            top_k=5,
        )
        if code_context:
            parts.append("## Relevant Code")
            for result in code_context:
                parts.append(f"### {result.repo}/{result.file_path}")
                parts.append(f"```\n{result.content}\n```")
            parts.append("")
        elif self._qdrant is not None:
            # Qdrant available but no code indexed
            parts.append("Note: No code has been indexed yet. Use /ingest to index repositories.")
            parts.append("")

        # Knowledge context
        knowledge_context = self._search_knowledge_base(task_description, top_k=5)
        if knowledge_context:
            parts.append("## Relevant Knowledge")
            for result in knowledge_context:
                summary = result.metadata.get("summary", result.metadata.get("content", ""))
                if summary:
                    parts.append(f"- {summary[:500]}")
            parts.append("")

        # Conversation memory
        memory = self._get_conversation_memory(agent_id, limit=10)
        if memory:
            parts.append("## Recent Conversation Context")
            for turn in memory:
                parts.append(f"- {turn.content[:300]}")
            parts.append("")

        # Task description
        parts.append(f"## Current Task\n{task_description}")

        return "\n".join(parts)

    def _search_code_index(
        self,
        query: str,
        repo_filter: list[str] | None = None,
        top_k: int = 5,
    ) -> list[CodeResult]:
        """Search Code_Index, filtering by assigned repos for dev agents."""
        if not self._qdrant or not self._embedding:
            return []

        try:
            # Check cache first
            embedding = None
            if self._cache:
                embedding = self._cache.get_embedding(query)

            if embedding is None:
                embedding = self._embedding.embed_text(query)
                if self._cache:
                    self._cache.put_embedding(query, embedding)

            # Search per repo if filter is set
            if repo_filter:
                results = []
                for repo in repo_filter:
                    repo_results = self._qdrant.search_code(
                        embedding, repo_filter=repo, top_k=top_k,
                    )
                    results.extend(repo_results)
                # Sort by score and take top_k
                results.sort(key=lambda r: r.score, reverse=True)
                return results[:top_k]
            else:
                return self._qdrant.search_code(embedding, top_k=top_k)
        except Exception as e:
            logger.warning("Code index search failed: %s", e)
            return []

    def _search_knowledge_base(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[KnowledgeResult]:
        """Search Knowledge_Base."""
        if not self._qdrant or not self._embedding:
            return []

        try:
            embedding = None
            if self._cache:
                embedding = self._cache.get_embedding(query)

            if embedding is None:
                embedding = self._embedding.embed_text(query)
                if self._cache:
                    self._cache.put_embedding(query, embedding)

            return self._qdrant.search_knowledge(embedding, top_k=top_k)
        except Exception as e:
            logger.warning("Knowledge base search failed: %s", e)
            return []

    def _get_conversation_memory(self, agent_id: str, limit: int = 10) -> list:
        """Get recent conversation turns for an agent."""
        if not self._qdrant or not agent_id:
            return []

        try:
            return self._qdrant.get_recent_context(agent_id, limit=limit)
        except Exception as e:
            logger.warning("Conversation memory retrieval failed: %s", e)
            return []
