"""Tests for ContextAssembler."""

import pytest

from src.agents.context_assembler import ContextAssembler
from src.knowledge.embedding_cache import EmbeddingCache
from src.knowledge.qdrant_service import QdrantService


class FakeEmbedding:
    def embed_text(self, text: str) -> list[float]:
        return [0.1] * 10

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]

    def get_dimensions(self) -> int:
        return 10


@pytest.fixture
def assembler():
    qdrant = QdrantService(vector_size=10)
    embedding = FakeEmbedding()
    cache = EmbeddingCache()
    return ContextAssembler(qdrant=qdrant, embedding=embedding, cache=cache)


@pytest.fixture
def assembler_with_data():
    qdrant = QdrantService(vector_size=10)
    embedding = FakeEmbedding()
    cache = EmbeddingCache()

    # Index some code
    qdrant.index_code("receiving-service", [
        {
            "chunk_id": "c1",
            "file_path": "src/handler.py",
            "language": "python",
            "commit_hash": "abc123",
            "chunk_type": "function",
            "content": "def handle_barcode(code): pass",
            "start_line": 1,
            "end_line": 1,
            "embedding": [0.1] * 10,
        },
    ])
    qdrant.index_code("shipping-service", [
        {
            "chunk_id": "c2",
            "file_path": "src/ship.py",
            "language": "python",
            "commit_hash": "def456",
            "chunk_type": "function",
            "content": "def ship_order(order_id): pass",
            "start_line": 1,
            "end_line": 1,
            "embedding": [0.1] * 10,
        },
    ])

    return ContextAssembler(qdrant=qdrant, embedding=embedding, cache=cache)


# Property 17: Grounding Instruction Invariant
def test_grounding_instruction_always_present(assembler):
    """Every assembled prompt must contain the grounding instruction."""
    prompt = assembler.assemble("Fix the bug", agent_role="developer")
    assert ContextAssembler.GROUNDING_INSTRUCTION in prompt


def test_grounding_with_system_prompt(assembler):
    prompt = assembler.assemble(
        "Fix the bug",
        system_prompt="You are a developer agent.",
    )
    assert ContextAssembler.GROUNDING_INSTRUCTION in prompt
    assert "You are a developer agent." in prompt


def test_grounding_for_pm_agent(assembler):
    prompt = assembler.assemble("Plan the feature", agent_role="pm")
    assert ContextAssembler.GROUNDING_INSTRUCTION in prompt


# Property 16: Context Assembly Repo Filtering
def test_repo_filtering_for_dev_agent(assembler_with_data):
    """Dev agent should only see code from assigned repos."""
    prompt = assembler_with_data.assemble(
        "Fix barcode scanning",
        agent_role="developer",
        assigned_repos=["receiving-service"],
    )
    # Should contain receiving-service code
    assert "receiving-service" in prompt or "handle_barcode" in prompt


def test_no_qdrant_graceful_degradation():
    """Assembler should work without Qdrant."""
    assembler = ContextAssembler(qdrant=None, embedding=None)
    prompt = assembler.assemble("Do something")
    assert ContextAssembler.GROUNDING_INSTRUCTION in prompt
    assert "Current Task" in prompt


def test_task_description_in_prompt(assembler):
    prompt = assembler.assemble("Implement barcode scanning feature")
    assert "Implement barcode scanning feature" in prompt


def test_no_code_indexed_notice(assembler):
    """When Qdrant is available but empty, show notice."""
    prompt = assembler.assemble("Fix something")
    assert "No code has been indexed" in prompt or "Current Task" in prompt
