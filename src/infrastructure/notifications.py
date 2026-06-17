"""Pluggable notification delivery interface.

Provides NotificationProvider protocol and ConsoleLogProvider implementation.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class NotificationProvider(Protocol):
    """Pluggable notification delivery interface."""

    def notify(self, event_type: str, payload: dict[str, Any]) -> None:
        """Deliver a notification for an event."""
        ...


class ConsoleLogProvider:
    """Prints notifications to stdout in structured format."""

    def __init__(self) -> None:
        self._history: list[dict] = []

    def notify(self, event_type: str, payload: dict[str, Any]) -> None:
        entry = {
            "event_type": event_type,
            "payload": payload,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._history.append(entry)
        print(f"[NOTIFICATION] {event_type}: {json.dumps(payload, default=str)}")

    def get_history(self) -> list[dict]:
        return list(self._history)
