"""Pluggable communication channels for human interaction.

Provides CommunicationChannel protocol and CLIChannel implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ChannelResponse:
    """A response received from a communication channel."""

    message_id: str
    responder: str
    text: str
    channel_name: str


@runtime_checkable
class CommunicationChannel(Protocol):
    """Pluggable channel for delivering agent questions to humans."""

    def send_question(self, question: str, context: dict,
                      recipient: str, urgency: str) -> str:
        """Send a question. Returns message_id for tracking."""
        ...

    def poll_response(self, message_id: str) -> Optional[ChannelResponse]:
        """Poll for a response. Returns None if no response yet."""
        ...


class CLIChannel:
    """Interactive CLI channel — stores questions for CLI display."""

    def __init__(self) -> None:
        self._pending: dict[str, dict] = {}
        self._responses: dict[str, ChannelResponse] = {}

    def send_question(self, question: str, context: dict,
                      recipient: str, urgency: str) -> str:
        msg_id = str(uuid4())
        self._pending[msg_id] = {
            "question": question,
            "context": context,
            "recipient": recipient,
            "urgency": urgency,
        }
        logger.info("[CLI] Question for %s: %s", recipient, question)
        return msg_id

    def poll_response(self, message_id: str) -> Optional[ChannelResponse]:
        return self._responses.get(message_id)

    def submit_response(self, message_id: str, text: str, responder: str = "human") -> None:
        """Submit a response (called by CLI REPL)."""
        self._responses[message_id] = ChannelResponse(
            message_id=message_id,
            responder=responder,
            text=text,
            channel_name="cli",
        )
        self._pending.pop(message_id, None)

    def get_pending_questions(self) -> list[dict]:
        """Get all pending questions for CLI display."""
        return [{"message_id": k, **v} for k, v in self._pending.items()]
