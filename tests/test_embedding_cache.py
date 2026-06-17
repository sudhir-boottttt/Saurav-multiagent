"""Tests for EmbeddingCache."""

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.knowledge.embedding_cache import EmbeddingCache


def test_cache_miss():
    cache = EmbeddingCache()
    assert cache.get_embedding("hello") is None


def test_cache_hit():
    cache = EmbeddingCache()
    emb = [0.1, 0.2, 0.3]
    cache.put_embedding("hello", emb)
    assert cache.get_embedding("hello") == emb


# Property 30: Cache Consistency
@given(
    text=st.text(min_size=1, max_size=100),
    embedding=st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False), min_size=3, max_size=10),
)
@settings(max_examples=50)
def test_cache_consistency(text, embedding):
    """Cached value must be identical to what was stored."""
    cache = EmbeddingCache()
    cache.put_embedding(text, embedding)
    cached = cache.get_embedding(text)
    assert cached == embedding


def test_search_result_cache():
    cache = EmbeddingCache()
    results = [{"id": "1", "score": 0.9}]
    cache.put_search_result("query", "code_index", 5, None, results)
    cached = cache.get_search_result("query", "code_index", 5, None)
    assert cached == results


def test_search_result_miss():
    cache = EmbeddingCache()
    assert cache.get_search_result("query", "code_index", 5, None) is None


def test_ttl_expiry():
    cache = EmbeddingCache(embedding_ttl=0)  # Immediate expiry
    cache.put_embedding("hello", [0.1])
    # Entry should be expired immediately
    assert cache.get_embedding("hello") is None


def test_clear():
    cache = EmbeddingCache()
    cache.put_embedding("a", [0.1])
    cache.put_search_result("q", "c", 5, None, [{"x": 1}])
    cache.clear()
    assert cache.get_embedding("a") is None
    assert cache.get_search_result("q", "c", 5, None) is None


def test_stats():
    cache = EmbeddingCache()
    cache.put_embedding("a", [0.1])
    cache.get_embedding("a")  # hit
    cache.get_embedding("b")  # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["embedding_entries"] == 1


def test_invalidate_collection():
    cache = EmbeddingCache()
    cache.put_search_result("q1", "code", 5, None, [{"x": 1}])
    cache.put_search_result("q2", "code", 5, None, [{"x": 2}])
    count = cache.invalidate_collection("code")
    assert count == 2
    assert cache.get_search_result("q1", "code", 5, None) is None
