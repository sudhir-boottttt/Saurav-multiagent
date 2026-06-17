"""Code indexing pipeline using AST-based parsing.

Parses source files into scoped chunks (functions, classes, module-level),
embeds them, and stores in Qdrant Code_Index.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from typing import Optional

from src.core.models import CodeChunk
from src.knowledge.embedding import EmbeddingProvider
from src.knowledge.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

# Approximate tokens per character ratio
CHARS_PER_TOKEN = 4
DEFAULT_MAX_CHUNK_TOKENS = 512


class CodeIndexingPipeline:
    """Parses repos into AST chunks, embeds, and stores in Qdrant Code_Index."""

    def __init__(
        self,
        embedding_service: EmbeddingProvider,
        qdrant_service: QdrantService,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    ) -> None:
        self._embedding = embedding_service
        self._qdrant = qdrant_service
        self._max_chunk_tokens = max_chunk_tokens
        self._max_chunk_chars = max_chunk_tokens * CHARS_PER_TOKEN
        self._indexed_commits: dict[str, str] = {}

    def index_repository(
        self,
        repo_path: str,
        repo_name: str,
        language: str = "python",
        commit_hash: str = "HEAD",
    ) -> int:
        """Full index of a repository. Returns chunk count."""
        repo = Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"Repository path not found: {repo_path}")

        extensions = _language_extensions(language)
        files = []
        for ext in extensions:
            files.extend(repo.rglob(f"*{ext}"))

        total_chunks = 0
        for file_path in files:
            try:
                chunks = self.parse_file(str(file_path), language)
                if not chunks:
                    continue

                # Set repo and commit info
                rel_path = str(file_path.relative_to(repo))
                for chunk in chunks:
                    chunk.repo = repo_name
                    chunk.file_path = rel_path
                    chunk.commit_hash = commit_hash

                # Embed and store
                texts = [c.content for c in chunks]
                embeddings = self._embedding.embed_batch(texts)

                qdrant_chunks = []
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb
                    qdrant_chunks.append({
                        "chunk_id": chunk.chunk_id,
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "commit_hash": chunk.commit_hash,
                        "chunk_type": chunk.chunk_type,
                        "content": chunk.content,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "embedding": emb,
                    })

                indexed = self._qdrant.index_code(repo_name, qdrant_chunks)
                total_chunks += indexed
            except Exception as e:
                logger.warning("Failed to index %s: %s", file_path, e)

        self._indexed_commits[repo_name] = commit_hash
        logger.info("Indexed %d chunks from %s (%s)", total_chunks, repo_name, commit_hash)
        return total_chunks

    def incremental_reindex(
        self,
        repo_path: str,
        repo_name: str,
        language: str = "python",
        old_commit: str = "",
        new_commit: str = "HEAD",
    ) -> tuple[int, int]:
        """Re-index only changed files. Returns (added, removed) counts."""
        import subprocess

        repo = Path(repo_path)
        changed_files: list[str] | None = None

        # Try to get changed files via git diff
        if old_commit:
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-status", old_commit, new_commit],
                    cwd=str(repo), capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    changed_files = []
                    extensions = set(_language_extensions(language))
                    for line in result.stdout.strip().splitlines():
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            status, fpath = parts[0].strip(), parts[1].strip()
                            if any(fpath.endswith(ext) for ext in extensions):
                                changed_files.append((status, fpath))
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("git diff failed, falling back to full re-index")

        # If we couldn't determine changed files, do a full re-index
        if changed_files is None:
            removed = 0
            keys_to_remove = []
            for pid, data in list(self._qdrant._code_index.items()):
                if data["metadata"].get("repo") == repo_name:
                    keys_to_remove.append(pid)
            for pid in keys_to_remove:
                del self._qdrant._code_index[pid]
                removed += 1
            added = self.index_repository(repo_path, repo_name, language, new_commit)
            return added, removed

        # Incremental: only process changed files
        removed = 0
        added = 0

        for status, rel_path in changed_files:
            # Remove old chunks for this file
            keys_to_remove = []
            for pid, data in list(self._qdrant._code_index.items()):
                meta = data["metadata"]
                if meta.get("repo") == repo_name and meta.get("file_path") == rel_path:
                    keys_to_remove.append(pid)
            for pid in keys_to_remove:
                del self._qdrant._code_index[pid]
                removed += 1

            # Re-index if file still exists (not deleted)
            if status != "D":
                file_path = repo / rel_path
                if file_path.exists():
                    try:
                        chunks = self.parse_file(str(file_path), language)
                        if chunks:
                            for chunk in chunks:
                                chunk.repo = repo_name
                                chunk.file_path = rel_path
                                chunk.commit_hash = new_commit

                            texts = [c.content for c in chunks]
                            embeddings = self._embedding.embed_batch(texts)
                            qdrant_chunks = []
                            for chunk, emb in zip(chunks, embeddings):
                                chunk.embedding = emb
                                qdrant_chunks.append({
                                    "chunk_id": chunk.chunk_id,
                                    "file_path": chunk.file_path,
                                    "language": chunk.language,
                                    "commit_hash": chunk.commit_hash,
                                    "chunk_type": chunk.chunk_type,
                                    "content": chunk.content,
                                    "start_line": chunk.start_line,
                                    "end_line": chunk.end_line,
                                    "embedding": emb,
                                })
                            added += self._qdrant.index_code(repo_name, qdrant_chunks)
                    except Exception as e:
                        logger.warning("Failed to re-index %s: %s", rel_path, e)

        self._indexed_commits[repo_name] = new_commit
        logger.info("Incremental re-index for %s: +%d -%d", repo_name, added, removed)
        return added, removed

    def parse_file(self, file_path: str, language: str = "python") -> list[CodeChunk]:
        """Parse a single file into AST-based code chunks."""
        path = Path(file_path)
        if not path.exists():
            return []

        source = path.read_text(encoding="utf-8", errors="replace")
        if not source.strip():
            return []

        if language == "python":
            return self._parse_python(source, str(path))
        else:
            # Fallback: chunk by lines for unsupported languages
            return self._chunk_by_lines(source, str(path), language)

    def _parse_python(self, source: str, file_path: str) -> list[CodeChunk]:
        """Parse Python source using ast module."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            logger.warning("Syntax error in %s, falling back to line chunking", file_path)
            return self._chunk_by_lines(source, file_path, "python")

        lines = source.splitlines()
        chunks: list[CodeChunk] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._extract_node_chunk(node, lines, file_path, "function")
                if chunk:
                    chunks.append(chunk)
            elif isinstance(node, ast.ClassDef):
                chunk = self._extract_node_chunk(node, lines, file_path, "class")
                if chunk:
                    chunks.append(chunk)

        # If no functions/classes found, treat as module-level
        if not chunks and source.strip():
            chunks.append(self._make_chunk(source, file_path, "module", 1, len(lines)))

        return chunks

    def _extract_node_chunk(
        self, node: ast.AST, lines: list[str], file_path: str, chunk_type: str,
    ) -> Optional[CodeChunk]:
        """Extract a code chunk from an AST node."""
        start = node.lineno  # type: ignore[attr-defined]
        end = node.end_lineno or start  # type: ignore[attr-defined]
        content = "\n".join(lines[start - 1 : end])

        # Enforce max chunk size
        if len(content) > self._max_chunk_chars:
            content = content[: self._max_chunk_chars]

        if not content.strip():
            return None

        return self._make_chunk(content, file_path, chunk_type, start, end)

    def _chunk_by_lines(self, source: str, file_path: str, language: str) -> list[CodeChunk]:
        """Fallback: chunk source by line count."""
        lines = source.splitlines()
        chunks = []
        chunk_size = self._max_chunk_chars
        current = []
        current_start = 1

        for i, line in enumerate(lines, 1):
            current.append(line)
            if len("\n".join(current)) >= chunk_size:
                content = "\n".join(current)
                chunks.append(self._make_chunk(content, file_path, "module", current_start, i))
                current = []
                current_start = i + 1

        if current:
            content = "\n".join(current)
            chunks.append(self._make_chunk(content, file_path, "module", current_start, len(lines)))

        return chunks

    def _make_chunk(
        self, content: str, file_path: str, chunk_type: str,
        start_line: int, end_line: int,
    ) -> CodeChunk:
        """Create a CodeChunk model instance."""
        return CodeChunk(
            repo="pending",
            file_path=file_path,
            chunk_type=chunk_type,
            language="python",
            content=content[:self._max_chunk_chars],
            start_line=max(1, start_line),
            end_line=max(1, end_line),
            commit_hash="pending",
        )

    def get_last_indexed_commit(self, repo_name: str) -> Optional[str]:
        """Get the last indexed commit hash for a repo."""
        return self._indexed_commits.get(repo_name)


def _language_extensions(language: str) -> list[str]:
    """Map language name to file extensions."""
    mapping = {
        "python": [".py"],
        "java": [".java"],
        "javascript": [".js", ".jsx"],
        "typescript": [".ts", ".tsx"],
        "go": [".go"],
    }
    return mapping.get(language, [f".{language}"])
