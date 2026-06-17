"""Human Interaction Point for workflow pauses requiring human input.

Delivers questions to all configured channels, polls for responses,
and acknowledges on other channels when answered.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from src.core.enums import HIPUrgency
from src.core.models import HIPRecord
from src.infrastructure.channels import CommunicationChannel, ChannelResponse

if TYPE_CHECKING:
    from src.infrastructure.storage import StorageBackend

logger = logging.getLogger(__name__)


class HumanInteractionPoint:
    """Manages workflow pauses for human input."""

    def __init__(
        self,
        channels: dict[str, CommunicationChannel] | None = None,
        storage: Optional[StorageBackend] = None,
        timeout_hours: int = 24,
    ) -> None:
        self._channels = channels or {}
        self._storage = storage
        self._timeout_hours = timeout_hours
        self._requests: dict[str, HIPRecord] = {}
        self._message_ids: dict[str, dict[str, str]] = {}  # hip_id -> {channel: msg_id}

    def request_input(
        self,
        question: str,
        context: dict,
        urgency: str,
        agent_id: str,
        task_id: str,
    ) -> HIPRecord:
        """Deliver question to all channels, return HIP for tracking."""
        now = datetime.now(UTC)
        hip = HIPRecord(
            hip_id=str(uuid4()),
            question=question,
            context=context,
            urgency=HIPUrgency(urgency) if urgency in [e.value for e in HIPUrgency] else HIPUrgency.NORMAL,
            agent_id=agent_id,
            task_id=task_id,
            created_at=now,
            timeout_at=now + timedelta(hours=self._timeout_hours),
        )
        self._requests[hip.hip_id] = hip

        # Send to all channels
        msg_ids: dict[str, str] = {}
        for name, channel in self._channels.items():
            try:
                msg_id = channel.send_question(question, context, agent_id, urgency)
                msg_ids[name] = msg_id
            except Exception as e:
                logger.warning("Failed to send to channel %s: %s", name, e)
        self._message_ids[hip.hip_id] = msg_ids

        if self._storage:
            self._storage.save("hip_records", hip.hip_id, hip.model_dump(mode="json"))

        logger.info("HIP %s created for agent %s: %s", hip.hip_id, agent_id, question[:80])
        return hip

    def check_response(self, hip_id: str) -> Optional[str]:
        """Poll all channels for a response."""
        hip = self._requests.get(hip_id)
        if not hip or hip.response is not None:
            return hip.response if hip else None

        msg_ids = self._message_ids.get(hip_id, {})
        for channel_name, msg_id in msg_ids.items():
            channel = self._channels.get(channel_name)
            if not channel:
                continue
            response = channel.poll_response(msg_id)
            if response:
                hip.response = response.text
                hip.responded_at = datetime.now(UTC)
                hip.responded_via = channel_name
                self.acknowledge_on_other_channels(hip_id, channel_name)
                if self._storage:
                    self._storage.save("hip_records", hip.hip_id, hip.model_dump(mode="json"))
                return response.text

        return None

    def acknowledge_on_other_channels(self, hip_id: str, responded_via: str) -> None:
        """Notify other channels that the question was answered."""
        msg_ids = self._message_ids.get(hip_id, {})
        for channel_name in msg_ids:
            if channel_name != responded_via:
                logger.info("HIP %s answered via %s (notifying %s)", hip_id, responded_via, channel_name)

    def get_pending(self) -> list[HIPRecord]:
        """Get all pending (unanswered) HIP requests."""
        return [h for h in self._requests.values() if h.response is None]
