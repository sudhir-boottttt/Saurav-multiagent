"""Knowledge ingestion pipeline for non-code documents.

Ingests .md, .txt, .json, .yaml files into Qdrant Knowledge_Base.
Chunks capped at 512 tokens.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.knowledge.embedding import EmbeddingProvider
from src.knowledge.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
DEFAULT_MAX_CHUNK_TOKENS = 512
SUPPORTED_FORMATS = {".md", ".txt", ".json", ".yaml", ".yml"}


class KnowledgeIngestionPipeline:
    """Ingests non-code documents into Qdrant Knowledge_Base."""

    def __init__(
        self,
        embedding_service: EmbeddingProvider,
        qdrant_service: QdrantService,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
        max_document_size: int = 1_048_576,
    ) -> None:
        self._embedding = embedding_service
        self._qdrant = qdrant_service
        self._max_chunk_chars = max_chunk_tokens * CHARS_PER_TOKEN
        self._max_doc_size = max_document_size

    def ingest_file(self, file_path: str) -> int:
        """Ingest a single file. Returns chunk count."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix not in SUPPORTED_FORMATS:
            logger.warning("Unsupported format: %s", path.suffix)
            return 0
        if path.stat().st_size > self._max_doc_size:
            logger.warning("File too large: %s (%d bytes)", file_path, path.stat().st_size)
            return 0

        content = path.read_text(encoding="utf-8", errors="replace")
        chunks = self._chunk_text(content)
        if not chunks:
            return 0

        embeddings = self._embedding.embed_batch(chunks)
        count = 0
        for chunk, emb in zip(chunks, embeddings):
            self._qdrant.store_knowledge(emb, {
                "source_file": str(path),
                "content": chunk,
                "type": "document",
                "format": path.suffix,
            })
            count += 1

        logger.info("Ingested %d chunks from %s", count, file_path)
        return count

    def ingest_directory(self, dir_path: str) -> int:
        """Ingest all supported files in a directory. Returns total chunk count."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        total = 0
        for fp in sorted(path.rglob("*")):
            if fp.is_file() and fp.suffix in SUPPORTED_FORMATS:
                try:
                    total += self.ingest_file(str(fp))
                except Exception as e:
                    logger.warning("Failed to ingest %s: %s", fp, e)
        return total

    def reingest_file(self, file_path: str) -> int:
        """Re-ingest a file, replacing old chunks. Returns new chunk count."""
        self._qdrant.delete_knowledge_by_source(str(Path(file_path)))
        return self.ingest_file(file_path)

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks capped at max_chunk_chars."""
        if not text.strip():
            return []

        # Try paragraph-based splitting first
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= self._max_chunk_chars:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    chunks.append(current.strip())
                if len(para) > self._max_chunk_chars:
                    # Split long paragraph by lines
                    for i in range(0, len(para), self._max_chunk_chars):
                        chunks.append(para[i:i + self._max_chunk_chars].strip())
                    current = ""
                else:
                    current = para

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if c]
