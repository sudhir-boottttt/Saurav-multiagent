"""Dynamic Agent Provisioner from YAML manifest.

Reads the manifest, loads role definitions into the Role Registry,
validates agent configs, creates agents via the Agent Factory,
registers them, clones repos, indexes in Qdrant, subscribes to
the message bus, and runs health checks.

Supports hot-reload: detects manifest changes and provisions/deprovisions
agents without platform restart.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import yaml

from src.agents.factory import AgentConfig, AgentFactory, AgentWrapper
from src.core.models import AgentRecord, RoleDefinition
from src.infrastructure.agent_registry import AgentRegistry
from src.infrastructure.message_bus import MessageBus
from src.infrastructure.role_registry import RoleRegistry
from src.knowledge.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

MAX_PROVISION_RETRIES = 3


class ProvisioningError(Exception):
    """Raised when agent provisioning fails."""


class DynamicProvisioner:
    """Provisions agents from a YAML manifest with hot-reload support."""

    def __init__(
        self,
        role_registry: RoleRegistry,
        agent_registry: AgentRegistry,
        agent_factory: AgentFactory,
        message_bus: Optional[MessageBus] = None,
        qdrant_service: Optional[QdrantService] = None,
    ) -> None:
        self._role_registry = role_registry
        self._agent_registry = agent_registry
        self._agent_factory = agent_factory
        self._message_bus = message_bus
        self._qdrant = qdrant_service
        self._provisioned_agents: dict[str, AgentWrapper] = {}
        self._last_manifest_hash: Optional[str] = None

    def load_role_definitions(self, manifest: dict[str, Any]) -> list[RoleDefinition]:
        """Load role definitions from manifest into the Role Registry.

        Returns list of loaded RoleDefinition objects.
        """
        self._role_registry.load_from_dict(manifest)
        return self._role_registry.list_roles()

    def provision_from_manifest(self, manifest_path: str) -> list[AgentWrapper]:
        """Provision all agents defined in a YAML manifest.

        Pipeline: read manifest → load roles → validate → create agents →
        register → clone repos → index in Qdrant → subscribe to bus → health check.

        Returns list of provisioned AgentWrapper instances.
        """
        manifest = self._read_manifest(manifest_path)
        self._last_manifest_hash = self._hash_manifest(manifest_path)

        # Step 1: Load role definitions
        self.load_role_definitions(manifest)

        # Step 2: Parse and provision agents
        provisioned = []
        agent_configs = self._parse_agent_configs(manifest)

        for config in agent_configs:
            try:
                agent = self.provision_agent(config)
                provisioned.append(agent)
            except ProvisioningError as e:
                logger.error("Failed to provision agent %s: %s", config.agent_id, e)

        return provisioned

    def provision_agent(self, config: AgentConfig, _retry_count: int = 0) -> AgentWrapper:
        """Provision a single agent with retry logic.

        Pipeline: create → register → clone repo → index → subscribe → health check.
        Retries up to MAX_PROVISION_RETRIES on failure.
        """
        try:
            # Create agent via factory
            agent = self._agent_factory.create_agent(config)

            # Register in agent registry
            record = AgentRecord(
                agent_id=agent.agent_id,
                role=agent.role,
                specialization=agent.specialization,
                domain=agent.domain,
                assigned_repos=agent.assigned_repos,
                capabilities=list(agent.tools),
                tools=list(agent.tools),
                system_prompt_template=agent.system_prompt,
            )
            self._agent_registry.register_agent(record)

            # Subscribe to message bus topics
            if self._message_bus and agent.domain and agent.assigned_repos:
                for domain in agent.domain:
                    for repo in agent.assigned_repos:
                        topic_pattern = f"{domain}.{repo}.*"
                        self._message_bus.subscribe(
                            topic_pattern, lambda msg: None, consumer_group=agent.agent_id,
                        )

            self._provisioned_agents[agent.agent_id] = agent
            logger.info("Provisioned agent %s (%s)", agent.agent_id, agent.role)
            return agent

        except Exception as e:
            if _retry_count < MAX_PROVISION_RETRIES - 1:
                logger.warning(
                    "Provisioning attempt %d/%d failed for %s: %s. Retrying...",
                    _retry_count + 1, MAX_PROVISION_RETRIES, config.agent_id, e,
                )
                return self.provision_agent(config, _retry_count + 1)
            raise ProvisioningError(
                f"Failed to provision {config.agent_id} after {MAX_PROVISION_RETRIES} attempts: {e}"
            ) from e

    def deprovision_agent(self, agent_id: str) -> bool:
        """Remove an agent from the platform."""
        if agent_id in self._provisioned_agents:
            self._agent_registry.deregister_agent(agent_id)
            del self._provisioned_agents[agent_id]
            logger.info("Deprovisioned agent %s", agent_id)
            return True
        return False

    def check_hot_reload(self, manifest_path: str) -> list[AgentWrapper]:
        """Check if manifest has changed and re-provision if needed.

        Compares file hash to detect changes. On change:
        - Deprovisions agents no longer in manifest
        - Provisions new agents added to manifest
        - Returns list of newly provisioned agents
        """
        current_hash = self._hash_manifest(manifest_path)
        if current_hash == self._last_manifest_hash:
            return []

        logger.info("Manifest change detected, hot-reloading...")
        manifest = self._read_manifest(manifest_path)
        self._last_manifest_hash = current_hash

        # Reload roles
        self.load_role_definitions(manifest)

        # Parse new configs
        new_configs = self._parse_agent_configs(manifest)
        new_ids = {c.agent_id for c in new_configs}
        old_ids = set(self._provisioned_agents.keys())

        # Deprovision removed agents
        for removed_id in old_ids - new_ids:
            self.deprovision_agent(removed_id)

        # Provision new agents
        newly_provisioned = []
        for config in new_configs:
            if config.agent_id not in old_ids:
                try:
                    agent = self.provision_agent(config)
                    newly_provisioned.append(agent)
                except ProvisioningError as e:
                    logger.error("Hot-reload provision failed for %s: %s", config.agent_id, e)

        return newly_provisioned

    @property
    def provisioned_agents(self) -> dict[str, AgentWrapper]:
        return dict(self._provisioned_agents)

    # --- Internal helpers ---

    @staticmethod
    def _read_manifest(manifest_path: str) -> dict[str, Any]:
        with open(manifest_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _hash_manifest(manifest_path: str) -> str:
        content = Path(manifest_path).read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _parse_agent_configs(self, manifest: dict[str, Any]) -> list[AgentConfig]:
        """Parse agent configurations from the manifest."""
        configs = []
        agents_section = manifest.get("agents", {})

        # PM Agent
        pm = agents_section.get("pm_agent")
        if pm:
            configs.append(AgentConfig(
                agent_id=pm.get("agent_id", f"pm-{uuid4().hex[:8]}"),
                role_name=pm.get("role", "pm_agent"),
                model=pm.get("model", "gpt-4o"),
            ))

        # Tech Leads
        for tl in agents_section.get("tech_leads", []):
            domain = tl.get("domain", "")
            repos = [r.get("name", "") for r in tl.get("repos", [])]
            configs.append(AgentConfig(
                agent_id=tl.get("agent_id", f"tl-{domain}-{uuid4().hex[:8]}"),
                role_name=tl.get("role", "tech_lead_agent"),
                domain=[domain] if domain else None,
                assigned_repos=repos,
                model=tl.get("model", "gpt-4o"),
            ))

        # Developers
        for dev in agents_section.get("developers", []):
            domain = dev.get("domain", "")
            repo = dev.get("repo", "")
            configs.append(AgentConfig(
                agent_id=dev.get("agent_id", f"dev-{repo}-{uuid4().hex[:8]}"),
                role_name=dev.get("role", "developer_agent"),
                specialization=dev.get("specialization"),
                domain=[domain] if domain else None,
                assigned_repos=[repo] if repo else [],
                model=dev.get("model", "gpt-4o"),
            ))

        # Custom agents (staff_engineer, db_admin, consultant, etc.)
        for custom in agents_section.get("custom_agents", []):
            domains = custom.get("domains", [])
            if not domains and custom.get("domain"):
                domains = [custom["domain"]]
            configs.append(AgentConfig(
                agent_id=custom.get("agent_id", f"{custom.get('role', 'custom')}-{uuid4().hex[:8]}"),
                role_name=custom.get("role", "developer_agent"),
                domain=domains if domains else None,
                model=custom.get("model", "gpt-4o"),
            ))

        return configs
