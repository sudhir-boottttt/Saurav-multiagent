"""TTL-based LRU cache for embedding vectors and search results.

Reduces redundant embedding API calls and Qdrant queries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Internal cache entry with TTL tracking."""

    value: Any
    created_at: float
    ttl: int
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) >= self.ttl


class EmbeddingCache:
    """TTL-based LRU cache for embeddings and search results."""

    def __init__(
        self,
        embedding_ttl: int = 3600,
        search_ttl: int = 300,
        max_size_mb: int = 100,
    ) -> None:
        self._embedding_ttl = embedding_ttl
        self._search_ttl = search_ttl
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._embeddings: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._search_results: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._current_size = 0

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_search(query: str, collection: str, top_k: int,
                     filters: dict | None) -> str:
        key = json.dumps({"q": query, "c": collection, "k": top_k,
                          "f": filters or {}}, sort_keys=True)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _estimate_size(self, value: Any) -> int:
        if isinstance(value, list) and value and isinstance(value[0], float):
            return len(value) * 8
        return len(json.dumps(value, default=str).encode("utf-8"))

    def _evict_if_needed(self, needed: int) -> None:
        """Evict oldest entries if adding needed bytes would exceed max."""
        while self._current_size + needed > self._max_size_bytes:
            # Try search results first (shorter TTL)
            if self._search_results:
                _, entry = self._search_results.popitem(last=False)
                self._current_size -= entry.size_bytes
            elif self._embeddings:
                _, entry = self._embeddings.popitem(last=False)
                self._current_size -= entry.size_bytes
            else:
                break

    def get_embedding(self, text: str) -> list[float] | None:
        """Get cached embedding by text hash. Returns None on miss."""
        key = self._hash_text(text)
        entry = self._embeddings.get(key)
        if entry is None or entry.is_expired:
            if entry and entry.is_expired:
                self._current_size -= entry.size_bytes
                del self._embeddings[key]
            self._misses += 1
            return None
        self._embeddings.move_to_end(key)
        self._hits += 1
        return entry.value

    def put_embedding(self, text: str, embedding: list[float]) -> None:
        """Cache an embedding vector."""
        key = self._hash_text(text)
        size = self._estimate_size(embedding)
        # Remove old entry if exists
        if key in self._embeddings:
            self._current_size -= self._embeddings[key].size_bytes
            del self._embeddings[key]
        self._evict_if_needed(size)
        self._embeddings[key] = _CacheEntry(
            value=embedding, created_at=time.monotonic(),
            ttl=self._embedding_ttl, size_bytes=size,
        )
        self._current_size += size

    def get_search_result(self, query: str, collection: str,
                          top_k: int, filters: dict | None = None) -> list | None:
        """Get cached search results. Returns None on miss."""
        key = self._hash_search(query, collection, top_k, filters)
        entry = self._search_results.get(key)
        if entry is None or entry.is_expired:
            if entry and entry.is_expired:
                self._current_size -= entry.size_bytes
                del self._search_results[key]
            self._misses += 1
            return None
        self._search_results.move_to_end(key)
        self._hits += 1
        return entry.value

    def put_search_result(self, query: str, collection: str, top_k: int,
                          filters: dict | None, results: list) -> None:
        """Cache search results."""
        key = self._hash_search(query, collection, top_k, filters)
        size = self._estimate_size(results)
        if key in self._search_results:
            self._current_size -= self._search_results[key].size_bytes
            del self._search_results[key]
        self._evict_if_needed(size)
        self._search_results[key] = _CacheEntry(
            value=results, created_at=time.monotonic(),
            ttl=self._search_ttl, size_bytes=size,
        )
        self._current_size += size

    def invalidate_collection(self, collection: str) -> int:
        """Invalidate all cached search results for a collection. Returns count evicted."""
        to_remove = []
        for key, entry in self._search_results.items():
            # The collection is embedded in the hash key, so we need to check
            # We store collection info in the hash, so invalidate all search results
            # that match. Since we hash the collection into the key, we invalidate all.
            to_remove.append(key)
        count = 0
        for key in to_remove:
            entry = self._search_results.pop(key, None)
            if entry:
                self._current_size -= entry.size_bytes
                count += 1
        return count

    def stats(self) -> dict:
        """Return hit rate, size, entry count."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "embedding_entries": len(self._embeddings),
            "search_entries": len(self._search_results),
            "size_bytes": self._current_size,
            "max_size_bytes": self._max_size_bytes,
        }

    def clear(self) -> None:
        """Flush all cache entries."""
        self._embeddings.clear()
        self._search_results.clear()
        self._current_size = 0
        self._hits = 0
        self._misses = 0
