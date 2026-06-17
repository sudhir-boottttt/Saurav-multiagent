"""Tests for KnowledgeIngestionPipeline."""

import pytest

from src.knowledge.knowledge_ingestion import KnowledgeIngestionPipeline
from src.knowledge.qdrant_service import QdrantService


class FakeEmbedding:
    def embed_text(self, text: str) -> list[float]:
        return [0.1] * 10

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]

    def get_dimensions(self) -> int:
        return 10


@pytest.fixture
def pipeline():
    return KnowledgeIngestionPipeline(
        embedding_service=FakeEmbedding(),
        qdrant_service=QdrantService(vector_size=10),
    )


def test_ingest_markdown(pipeline, tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nSome content about barcode scanning.\n\nMore details here.")
    count = pipeline.ingest_file(str(f))
    assert count >= 1


def test_ingest_txt(pipeline, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Important note about the system.")
    count = pipeline.ingest_file(str(f))
    assert count >= 1


def test_ingest_unsupported_format(pipeline, tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    count = pipeline.ingest_file(str(f))
    assert count == 0


def test_ingest_nonexistent(pipeline):
    with pytest.raises(FileNotFoundError):
        pipeline.ingest_file("/nonexistent/file.md")


def test_ingest_directory(pipeline, tmp_path):
    (tmp_path / "a.md").write_text("Doc A content")
    (tmp_path / "b.txt").write_text("Doc B content")
    (tmp_path / "c.py").write_text("# not ingested")
    count = pipeline.ingest_directory(str(tmp_path))
    assert count >= 2


def test_reingest_file(pipeline, tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("Version 1")
    pipeline.ingest_file(str(f))
    f.write_text("Version 2 updated")
    count = pipeline.reingest_file(str(f))
    assert count >= 1


def test_empty_file(pipeline, tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("")
    count = pipeline.ingest_file(str(f))
    assert count == 0
