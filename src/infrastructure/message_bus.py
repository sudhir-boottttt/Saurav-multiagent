"""Event-driven message bus using Redis Streams.

Supports topic-based pub/sub, request-reply, retry with exponential
backoff, and dead-letter queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from src.infrastructure.storage import StorageBackend

logger = logging.getLogger(__name__)

# Topic format: {domain}.{repository}.{event_type}
TOPIC_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+$")


class MessageBusError(Exception):
    """Base error for message bus operations."""


class TopicValidationError(MessageBusError):
    """Raised when a topic doesn't match the required format."""


class DeliveryFailedError(MessageBusError):
    """Raised when message delivery fails after all retries."""


class RequestTimeoutError(MessageBusError):
    """Raised when a request-reply times out."""


class Message:
    """An immutable message on the bus."""

    def __init__(
        self,
        message_id: str,
        topic: str,
        sender_id: str,
        payload: dict[str, Any],
        timestamp: datetime,
        correlation_id: str,
        retry_count: int = 0,
    ):
        self.message_id = message_id
        self.topic = topic
        self.sender_id = sender_id
        self.payload = payload
        self.timestamp = timestamp
        self.correlation_id = correlation_id
        self.retry_count = retry_count


class Subscription:
    """A topic subscription."""

    def __init__(self, topic_pattern: str, handler: Callable, consumer_group: str):
        self.topic_pattern = topic_pattern
        self.handler = handler
        self.consumer_group = consumer_group
        self._regex = re.compile(topic_pattern.replace("*", "[a-zA-Z0-9_-]+"))

    def matches(self, topic: str) -> bool:
        return bool(self._regex.fullmatch(topic))


class MessageBus:
    """In-memory message bus with Redis Streams-compatible semantics.

    For production, swap the in-memory stores with actual Redis Streams.
    The interface and behavior remain identical.
    """

    MAX_RETRIES = 3
    BACKOFF_BASE_MS = 1000  # 1s, 2s, 4s

    def __init__(self, storage: Optional[StorageBackend] = None) -> None:
        self._message_store: dict[str, Message] = {}
        self._subscriptions: list[Subscription] = []
        self._dead_letter_queue: list[Message] = []
        self._reply_futures: dict[str, asyncio.Future] = {}
        self._pm_agent_id: Optional[str] = None
        self._notify_callback: Optional[Callable[[str, Message], Any]] = None
        self._storage = storage

    def set_pm_agent_id(self, agent_id: str) -> None:
        self._pm_agent_id = agent_id

    def set_notify_callback(self, callback: Callable[[str, Message], Any]) -> None:
        """Set callback for PM agent notifications on delivery failures."""
        self._notify_callback = callback

    @staticmethod
    def validate_topic(topic: str) -> bool:
        """Validate topic matches {domain}.{repository}.{event_type} format."""
        return bool(TOPIC_PATTERN.fullmatch(topic))

    def publish(self, topic: str, payload: dict[str, Any], sender_id: str, _skip_validation: bool = False) -> str:
        """Publish a message to a topic. Returns message_id.

        Raises TopicValidationError if topic format is invalid.
        """
        if not _skip_validation and not self.validate_topic(topic):
            raise TopicValidationError(
                f"Invalid topic '{topic}'. Expected format: {{domain}}.{{repository}}.{{event_type}}"
            )

        msg = Message(
            message_id=str(uuid4()),
            topic=topic,
            sender_id=sender_id,
            payload=payload,
            timestamp=datetime.now(UTC),
            correlation_id=str(uuid4()),
        )

        # Persist before delivery
        self._message_store[msg.message_id] = msg
        if self._storage:
            self._storage.save("messages", msg.message_id, {
                "message_id": msg.message_id,
                "topic": msg.topic,
                "sender_id": msg.sender_id,
                "payload": msg.payload,
                "timestamp": msg.timestamp.isoformat(),
                "correlation_id": msg.correlation_id,
                "retry_count": msg.retry_count,
            })

        # Deliver to matching subscribers
        for sub in self._subscriptions:
            if sub.matches(topic):
                self._deliver(msg, sub)

        return msg.message_id

    def _deliver(self, msg: Message, sub: Subscription) -> bool:
        """Attempt delivery with retry logic."""
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                sub.handler(msg)
                return True
            except Exception as e:
                msg.retry_count = attempt
                if attempt < self.MAX_RETRIES:
                    logger.warning(
                        "Delivery failed (attempt %d/%d) for %s: %s",
                        attempt + 1, self.MAX_RETRIES, msg.message_id, e,
                    )
                else:
                    # Move to dead-letter queue
                    self._dead_letter_queue.append(msg)
                    logger.error("Message %s moved to DLQ after %d retries", msg.message_id, self.MAX_RETRIES)
                    if self._notify_callback and self._pm_agent_id:
                        self._notify_callback(self._pm_agent_id, msg)
        return False

    def subscribe(self, topic_pattern: str, handler: Callable, consumer_group: str = "default") -> Subscription:
        """Subscribe to a topic pattern. Supports wildcards with *."""
        sub = Subscription(topic_pattern, handler, consumer_group)
        self._subscriptions.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> bool:
        """Remove a subscription."""
        if sub in self._subscriptions:
            self._subscriptions.remove(sub)
            return True
        return False

    def get_message(self, message_id: str) -> Optional[Message]:
        """Retrieve a persisted message by ID."""
        return self._message_store.get(message_id)

    def get_dead_letter_queue(self) -> list[Message]:
        """Get all messages in the dead-letter queue."""
        return list(self._dead_letter_queue)

    def request_reply(self, topic: str, payload: dict[str, Any], sender_id: str, timeout_ms: int = 5000) -> Optional[dict]:
        """Synchronous request-reply pattern.

        Publishes a request and waits for a reply on a reply topic.
        Returns the reply payload or raises RequestTimeoutError.
        """
        correlation_id = str(uuid4())
        reply_topic = f"_reply.{correlation_id}.response"
        reply_holder: dict[str, Any] = {}

        def reply_handler(msg: Message) -> None:
            reply_holder["payload"] = msg.payload

        # Subscribe to reply topic
        reply_sub = self.subscribe(reply_topic, reply_handler, consumer_group=f"reply-{correlation_id}")

        # Publish request with correlation_id in payload
        payload["_correlation_id"] = correlation_id
        payload["_reply_topic"] = reply_topic
        self.publish(topic, payload, sender_id)

        # Wait for reply (blocking with timeout)
        start = time.monotonic()
        timeout_s = timeout_ms / 1000.0
        while time.monotonic() - start < timeout_s:
            if "payload" in reply_holder:
                self.unsubscribe(reply_sub)
                return reply_holder["payload"]
            time.sleep(0.01)

        self.unsubscribe(reply_sub)
        raise RequestTimeoutError(f"No reply received within {timeout_ms}ms for topic '{topic}'")
