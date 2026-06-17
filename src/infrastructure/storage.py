"""Pluggable persistence layer with StorageBackend protocol.

Provides SQLiteBackend and FilesystemBackend implementations.
All platform state flows through this interface.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class StorageBackend(Protocol):
    """Pluggable persistence interface for all platform state."""

    def save(self, collection: str, key: str, data: dict[str, Any]) -> None:
        """Persist data under collection/key."""
        ...

    def load(self, collection: str, key: str) -> Optional[dict[str, Any]]:
        """Load data by collection/key. Returns None if not found."""
        ...

    def query(self, collection: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Query a collection with field-level filters."""
        ...

    def delete(self, collection: str, key: str) -> None:
        """Delete a record by collection/key."""
        ...

    def list_keys(self, collection: str) -> list[str]:
        """List all keys in a collection."""
        ...


class SQLiteBackend:
    """SQLite-based storage backend.

    Stores all data in {base_path}/.platform-state/platform.db.
    Thread-safe via connection-per-thread with WAL mode.
    """

    def __init__(self, base_path: str = "/tmp/agent-repos/") -> None:
        state_dir = Path(base_path) / ".platform-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(state_dir / "platform.db")
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kv_store ("
            "  collection TEXT NOT NULL,"
            "  key TEXT NOT NULL,"
            "  data TEXT NOT NULL,"
            "  PRIMARY KEY (collection, key)"
            ")"
        )
        conn.commit()

    def save(self, collection: str, key: str, data: dict[str, Any]) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO kv_store (collection, key, data) VALUES (?, ?, ?)",
            (collection, key, json.dumps(data, default=str)),
        )
        conn.commit()

    def load(self, collection: str, key: str) -> Optional[dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data FROM kv_store WHERE collection = ? AND key = ?",
            (collection, key),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def query(self, collection: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT data FROM kv_store WHERE collection = ?",
            (collection,),
        ).fetchall()
        results = []
        for (raw,) in rows:
            record = json.loads(raw)
            if all(record.get(k) == v for k, v in filters.items()):
                results.append(record)
        return results

    def delete(self, collection: str, key: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM kv_store WHERE collection = ? AND key = ?",
            (collection, key),
        )
        conn.commit()

    def list_keys(self, collection: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT key FROM kv_store WHERE collection = ?",
            (collection,),
        ).fetchall()
        return [r[0] for r in rows]


class FilesystemBackend:
    """Filesystem-based storage backend.

    Stores data as JSON files under {base_path}/.platform-state/{collection}/{key}.json.
    """

    def __init__(self, base_path: str = "/tmp/agent-repos/") -> None:
        self._state_dir = Path(base_path) / ".platform-state"
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _collection_dir(self, collection: str) -> Path:
        d = self._state_dir / collection
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _file_path(self, collection: str, key: str) -> Path:
        return self._collection_dir(collection) / f"{key}.json"

    def save(self, collection: str, key: str, data: dict[str, Any]) -> None:
        path = self._file_path(collection, key)
        path.write_text(json.dumps(data, default=str), encoding="utf-8")

    def load(self, collection: str, key: str) -> Optional[dict[str, Any]]:
        path = self._file_path(collection, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def query(self, collection: str, filters: dict[str, Any]) -> list[dict[str, Any]]:
        results = []
        col_dir = self._collection_dir(collection)
        for fp in col_dir.glob("*.json"):
            record = json.loads(fp.read_text(encoding="utf-8"))
            if all(record.get(k) == v for k, v in filters.items()):
                results.append(record)
        return results

    def delete(self, collection: str, key: str) -> None:
        path = self._file_path(collection, key)
        if path.exists():
            path.unlink()

    def list_keys(self, collection: str) -> list[str]:
        col_dir = self._collection_dir(collection)
        return [fp.stem for fp in col_dir.glob("*.json")]
