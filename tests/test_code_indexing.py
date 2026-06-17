"""Tests for CodeIndexingPipeline."""

import tempfile
from pathlib import Path

import pytest

from src.knowledge.code_indexing import CodeIndexingPipeline
from src.knowledge.qdrant_service import QdrantService


class FakeEmbedding:
    """Fake embedding provider for testing."""

    def embed_text(self, text: str) -> list[float]:
        return [0.1] * 10

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]

    def get_dimensions(self) -> int:
        return 10


@pytest.fixture
def pipeline():
    return CodeIndexingPipeline(
        embedding_service=FakeEmbedding(),
        qdrant_service=QdrantService(vector_size=10),
    )


def test_parse_python_file(pipeline, tmp_path):
    """Parse a Python file into function and class chunks."""
    code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"

class Greeter:
    """A greeter class."""
    def greet(self):
        return "hi"
'''
    f = tmp_path / "test.py"
    f.write_text(code)
    chunks = pipeline.parse_file(str(f), "python")
    assert len(chunks) >= 2  # function + class
    types = {c.chunk_type for c in chunks}
    assert "function" in types
    assert "class" in types


def test_parse_empty_file(pipeline, tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("")
    chunks = pipeline.parse_file(str(f), "python")
    assert chunks == []


def test_parse_nonexistent_file(pipeline):
    chunks = pipeline.parse_file("/nonexistent/file.py", "python")
    assert chunks == []


# Property 4: Code Chunk Size Invariant
def test_chunk_size_invariant(pipeline, tmp_path):
    """Every chunk should be at most 512 tokens (~2048 chars)."""
    # Create a file with a very long function
    lines = ["def big_func():"] + [f"    x_{i} = {i}" for i in range(1000)]
    f = tmp_path / "big.py"
    f.write_text("\n".join(lines))
    chunks = pipeline.parse_file(str(f), "python")
    max_chars = 512 * 4  # CHARS_PER_TOKEN = 4
    for chunk in chunks:
        assert len(chunk.content) <= max_chars


def test_index_repository(pipeline, tmp_path):
    """Index a repository and verify chunks are stored."""
    (tmp_path / "main.py").write_text('def main():\n    return 42\n')
    (tmp_path / "utils.py").write_text('def helper():\n    pass\n')

    count = pipeline.index_repository(str(tmp_path), "test-repo", "python", "abc123")
    assert count >= 2


def test_index_nonexistent_repo(pipeline):
    with pytest.raises(FileNotFoundError):
        pipeline.index_repository("/nonexistent", "repo", "python")


def test_get_last_indexed_commit(pipeline, tmp_path):
    (tmp_path / "main.py").write_text('x = 1\n')
    pipeline.index_repository(str(tmp_path), "repo", "python", "commit123")
    assert pipeline.get_last_indexed_commit("repo") == "commit123"
    assert pipeline.get_last_indexed_commit("unknown") is None


def test_incremental_reindex(pipeline, tmp_path):
    (tmp_path / "main.py").write_text('def f():\n    return 1\n')
    pipeline.index_repository(str(tmp_path), "repo", "python", "c1")

    # Modify file
    (tmp_path / "main.py").write_text('def f():\n    return 2\n')
    added, removed = pipeline.incremental_reindex(str(tmp_path), "repo", "python", "c1", "c2")
    assert added >= 1
    assert removed >= 1


def test_syntax_error_fallback(pipeline, tmp_path):
    """Files with syntax errors should fall back to line chunking."""
    f = tmp_path / "bad.py"
    f.write_text("def broken(\n    pass\n")
    chunks = pipeline.parse_file(str(f), "python")
    assert len(chunks) >= 1  # Should still produce chunks via fallback
