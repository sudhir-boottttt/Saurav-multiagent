"""Agent Registry with health monitoring and auto-restart.

Manages agent lifecycle: registration, status tracking, heartbeat
monitoring, and automatic restart on failure.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Optional

from src.core.enums import AgentStatus
from src.core.models import AgentRecord

if TYPE_CHECKING:
    from src.infrastructure.storage import StorageBackend

logger = logging.getLogger(__name__)


class AgentFilter:
    """Filter criteria for listing agents."""

    def __init__(
        self,
        role: Optional[str] = None,
        domain: Optional[str] = None,
        status: Optional[AgentStatus] = None,
        specialization: Optional[str] = None,
    ):
        self.role = role
        self.domain = domain
        self.status = status
        self.specialization = specialization


class AgentRegistry:
    """Central registry for all agent instances.

    Tracks agents, their status, and provides health monitoring.
    Uses in-memory storage (can be backed by PostgreSQL + Redis in prod).
    """

    UNHEALTHY_THRESHOLD_SECONDS = 30
    RESTART_THRESHOLD_SECONDS = 120

    def __init__(self, storage: Optional[StorageBackend] = None) -> None:
        self._agents: dict[str, AgentRecord] = {}
        self._restart_callback: Optional[Callable[[str], Any]] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._storage = storage

    def set_restart_callback(self, callback: Callable[[str], Any]) -> None:
        """Set the callback invoked when an agent needs restart."""
        self._restart_callback = callback

    # --- CRUD operations ---

    def register_agent(self, agent: AgentRecord) -> AgentRecord:
        """Register a new agent. Raises ValueError on duplicate ID or missing role."""
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent '{agent.agent_id}' already registered")
        if not agent.role:
            raise ValueError("Agent must have a non-empty role")
        agent.status = AgentStatus.STARTING
        agent.created_at = datetime.now(UTC)
        agent.last_heartbeat = datetime.now(UTC)
        self._agents[agent.agent_id] = agent
        if self._storage:
            self._storage.save("agents", agent.agent_id, agent.model_dump(mode="json"))
        logger.info("Registered agent %s (role=%s)", agent.agent_id, agent.role)
        return agent

    def deregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the registry. Returns True if found and removed."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            if self._storage:
                self._storage.delete("agents", agent_id)
            logger.info("Deregistered agent %s", agent_id)
            return True
        return False

    def get_agent(self, agent_id: str) -> AgentRecord:
        """Get an agent by ID. Raises KeyError if not found."""
        if agent_id not in self._agents:
            raise KeyError(f"Agent '{agent_id}' not found")
        return self._agents[agent_id]

    def list_agents(self, filter: Optional[AgentFilter] = None) -> list[AgentRecord]:
        """List agents, optionally filtered."""
        agents = list(self._agents.values())
        if filter:
            if filter.role:
                agents = [a for a in agents if a.role == filter.role]
            if filter.domain:
                agents = [a for a in agents if a.domain and filter.domain in a.domain]
            if filter.status:
                agents = [a for a in agents if a.status == filter.status]
            if filter.specialization:
                agents = [a for a in agents if a.specialization == filter.specialization]
        return agents

    def update_status(self, agent_id: str, status: AgentStatus) -> AgentRecord:
        """Update an agent's status. Raises KeyError if not found."""
        agent = self.get_agent(agent_id)
        agent.status = status
        if status == AgentStatus.HEALTHY:
            agent.last_heartbeat = datetime.now(UTC)
        if self._storage:
            self._storage.save("agents", agent_id, agent.model_dump(mode="json"))
        logger.info("Agent %s status -> %s", agent_id, status.value)
        return agent

    def record_heartbeat(self, agent_id: str) -> AgentRecord:
        """Record a heartbeat from an agent, marking it healthy."""
        agent = self.get_agent(agent_id)
        agent.last_heartbeat = datetime.now(UTC)
        if agent.status != AgentStatus.HEALTHY:
            agent.status = AgentStatus.HEALTHY
        return agent

    # --- Health monitoring ---

    def health_check(self) -> dict[str, AgentStatus]:
        """Run a health check on all agents. Returns {agent_id: status}."""
        now = datetime.now(UTC)
        results: dict[str, AgentStatus] = {}
        for agent_id, agent in self._agents.items():
            elapsed = (now - agent.last_heartbeat).total_seconds()
            if agent.status == AgentStatus.STOPPED:
                results[agent_id] = AgentStatus.STOPPED
            elif elapsed > self.RESTART_THRESHOLD_SECONDS:
                agent.status = AgentStatus.UNHEALTHY
                results[agent_id] = AgentStatus.UNHEALTHY
            elif elapsed > self.UNHEALTHY_THRESHOLD_SECONDS:
                agent.status = AgentStatus.UNHEALTHY
                results[agent_id] = AgentStatus.UNHEALTHY
            else:
                results[agent_id] = agent.status
        return results

    def get_agents_needing_restart(self) -> list[str]:
        """Get agent IDs that have been unhealthy beyond the restart threshold."""
        now = datetime.now(UTC)
        restart_ids = []
        for agent_id, agent in self._agents.items():
            if agent.status == AgentStatus.STOPPED:
                continue
            elapsed = (now - agent.last_heartbeat).total_seconds()
            if elapsed > self.RESTART_THRESHOLD_SECONDS:
                restart_ids.append(agent_id)
        return restart_ids

    async def restart_agent(self, agent_id: str) -> bool:
        """Restart an agent and restore its context."""
        agent = self.get_agent(agent_id)
        logger.warning("Restarting agent %s (role=%s)", agent_id, agent.role)
        agent.status = AgentStatus.STARTING
        if self._restart_callback:
            await asyncio.coroutine(lambda: self._restart_callback(agent_id))() \
                if not asyncio.iscoroutinefunction(self._restart_callback) \
                else await self._restart_callback(agent_id)
        agent.status = AgentStatus.HEALTHY
        agent.last_heartbeat = datetime.now(UTC)
        return True

    async def start_health_monitor(self, interval_seconds: float = 15.0) -> None:
        """Start the background health monitoring loop."""
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval_seconds))

    async def stop_health_monitor(self) -> None:
        """Stop the background health monitoring loop."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_loop(self, interval: float) -> None:
        """Background loop that checks health and triggers restarts."""
        while True:
            try:
                self.health_check()
                for agent_id in self.get_agents_needing_restart():
                    try:
                        await self.restart_agent(agent_id)
                    except Exception:
                        logger.exception("Failed to restart agent %s", agent_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Health monitor error")
            await asyncio.sleep(interval)
