"""Tests for StorageBackend implementations."""

import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.infrastructure.storage import FilesystemBackend, SQLiteBackend, StorageBackend


@pytest.fixture
def sqlite_backend(tmp_path):
    return SQLiteBackend(base_path=str(tmp_path))


@pytest.fixture
def fs_backend(tmp_path):
    return FilesystemBackend(base_path=str(tmp_path))


# Property 7: StorageBackend Round-Trip
@given(
    collection=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnop"),
    key=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnop0123456789"),
    value=st.fixed_dictionaries({
        "name": st.text(min_size=1, max_size=50),
        "count": st.integers(min_value=0, max_value=10000),
    }),
)
@settings(max_examples=50)
def test_sqlite_round_trip(collection, key, value):
    """Save then load returns identical data for SQLiteBackend."""
    with tempfile.TemporaryDirectory() as tmp:
        backend = SQLiteBackend(base_path=tmp)
        backend.save(collection, key, value)
        loaded = backend.load(collection, key)
        assert loaded == value


@given(
    collection=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnop"),
    key=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnop0123456789"),
    value=st.fixed_dictionaries({
        "name": st.text(min_size=1, max_size=50),
        "count": st.integers(min_value=0, max_value=10000),
    }),
)
@settings(max_examples=50)
def test_filesystem_round_trip(collection, key, value):
    """Save then load returns identical data for FilesystemBackend."""
    with tempfile.TemporaryDirectory() as tmp:
        backend = FilesystemBackend(base_path=tmp)
        backend.save(collection, key, value)
        loaded = backend.load(collection, key)
        assert loaded == value


def test_sqlite_load_missing(sqlite_backend):
    assert sqlite_backend.load("col", "missing") is None


def test_sqlite_delete(sqlite_backend):
    sqlite_backend.save("col", "k1", {"a": 1})
    sqlite_backend.delete("col", "k1")
    assert sqlite_backend.load("col", "k1") is None


def test_sqlite_list_keys(sqlite_backend):
    sqlite_backend.save("col", "k1", {"a": 1})
    sqlite_backend.save("col", "k2", {"a": 2})
    keys = sqlite_backend.list_keys("col")
    assert sorted(keys) == ["k1", "k2"]


def test_sqlite_query(sqlite_backend):
    sqlite_backend.save("col", "k1", {"status": "active", "name": "a"})
    sqlite_backend.save("col", "k2", {"status": "inactive", "name": "b"})
    sqlite_backend.save("col", "k3", {"status": "active", "name": "c"})
    results = sqlite_backend.query("col", {"status": "active"})
    assert len(results) == 2


def test_fs_load_missing(fs_backend):
    assert fs_backend.load("col", "missing") is None


def test_fs_delete(fs_backend):
    fs_backend.save("col", "k1", {"a": 1})
    fs_backend.delete("col", "k1")
    assert fs_backend.load("col", "k1") is None


def test_fs_list_keys(fs_backend):
    fs_backend.save("col", "k1", {"a": 1})
    fs_backend.save("col", "k2", {"a": 2})
    keys = fs_backend.list_keys("col")
    assert sorted(keys) == ["k1", "k2"]


def test_fs_query(fs_backend):
    fs_backend.save("col", "k1", {"status": "active", "name": "a"})
    fs_backend.save("col", "k2", {"status": "inactive", "name": "b"})
    results = fs_backend.query("col", {"status": "active"})
    assert len(results) == 1


def test_protocol_compliance():
    """Both backends implement StorageBackend protocol."""
    with tempfile.TemporaryDirectory() as tmp:
        assert isinstance(SQLiteBackend(base_path=tmp), StorageBackend)
        assert isinstance(FilesystemBackend(base_path=tmp), StorageBackend)


def test_sqlite_overwrite(sqlite_backend):
    sqlite_backend.save("col", "k1", {"v": 1})
    sqlite_backend.save("col", "k1", {"v": 2})
    assert sqlite_backend.load("col", "k1") == {"v": 2}


def test_fs_overwrite(fs_backend):
    fs_backend.save("col", "k1", {"v": 1})
    fs_backend.save("col", "k1", {"v": 2})
    assert fs_backend.load("col", "k1") == {"v": 2}
