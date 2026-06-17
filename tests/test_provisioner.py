"""Tests for the Dynamic Agent Provisioner.

Covers end-to-end provisioning from manifest, hot-reload, and failure recovery.
"""

import os
import shutil
import tempfile

import pytest
import yaml

from src.agents.factory import AgentConfig, AgentFactory, AgentWrapper
from src.infrastructure.agent_registry import AgentRegistry
from src.infrastructure.message_bus import MessageBus
from src.infrastructure.provisioner import DynamicProvisioner, ProvisioningError
from src.infrastructure.role_registry import RoleRegistry
from src.knowledge.qdrant_service import QdrantService


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TEST_MANIFEST = os.path.join(FIXTURE_DIR, "test_manifest.yaml")


@pytest.fixture
def provisioner_deps():
    """Create all dependencies for the provisioner."""
    rr = RoleRegistry()
    ar = AgentRegistry()
    factory = AgentFactory(rr)
    bus = MessageBus()
    qdrant = QdrantService()
    return rr, ar, factory, bus, qdrant


@pytest.fixture
def provisioner(provisioner_deps):
    rr, ar, factory, bus, qdrant = provisioner_deps
    return DynamicProvisioner(rr, ar, factory, bus, qdrant)


class TestProvisionFromManifest:

    def test_provision_all_agents(self, provisioner):
        agents = provisioner.provision_from_manifest(TEST_MANIFEST)
        # Should provision: pm, 1 tech lead, 1 developer, 1 db_admin = 4
        assert len(agents) == 4
        roles = {a.role for a in agents}
        assert "pm_agent" in roles
        assert "tech_lead_agent" in roles
        assert "developer_agent" in roles
        assert "db_admin" in roles

    def test_roles_loaded_into_registry(self, provisioner, provisioner_deps):
        rr = provisioner_deps[0]
        provisioner.provision_from_manifest(TEST_MANIFEST)
        assert rr.get_role("pm_agent") is not None
        assert rr.get_role("tech_lead_agent") is not None
        assert rr.get_role("db_admin") is not None

    def test_agents_registered_in_registry(self, provisioner, provisioner_deps):
        ar = provisioner_deps[1]
        provisioner.provision_from_manifest(TEST_MANIFEST)
        # All provisioned agents should be in the agent registry
        for agent_id in provisioner.provisioned_agents:
            record = ar.get_agent(agent_id)
            assert record is not None

    def test_developer_has_specialization(self, provisioner):
        agents = provisioner.provision_from_manifest(TEST_MANIFEST)
        devs = [a for a in agents if a.role == "developer_agent"]
        assert len(devs) == 1
        assert devs[0].specialization == "backend"


class TestProvisionAgent:

    def test_provision_single_agent(self, provisioner):
        # Load roles first
        provisioner.provision_from_manifest(TEST_MANIFEST)
        config = AgentConfig(
            agent_id="dev-new-001",
            role_name="developer_agent",
            specialization="backend",
            domain=["inbound"],
            assigned_repos=["new-service"],
        )
        agent = provisioner.provision_agent(config)
        assert agent.agent_id == "dev-new-001"
        assert agent.role == "developer_agent"

    def test_deprovision_agent(self, provisioner):
        provisioner.provision_from_manifest(TEST_MANIFEST)
        agent_ids = list(provisioner.provisioned_agents.keys())
        assert len(agent_ids) > 0
        result = provisioner.deprovision_agent(agent_ids[0])
        assert result is True
        assert agent_ids[0] not in provisioner.provisioned_agents

    def test_deprovision_nonexistent_returns_false(self, provisioner):
        assert provisioner.deprovision_agent("nonexistent") is False


class TestHotReload:

    def test_hot_reload_detects_no_change(self, provisioner):
        provisioner.provision_from_manifest(TEST_MANIFEST)
        new_agents = provisioner.check_hot_reload(TEST_MANIFEST)
        assert len(new_agents) == 0

    def test_hot_reload_detects_new_agent(self, provisioner):
        provisioner.provision_from_manifest(TEST_MANIFEST)

        # Create modified manifest with an extra developer
        with open(TEST_MANIFEST) as f:
            manifest = yaml.safe_load(f)

        manifest["agents"]["developers"].append({
            "agent_id": "dev-new-hot-001",
            "role": "developer_agent",
            "specialization": "backend",
            "domain": "outbound",
            "repo": "picking-service",
            "model": "gpt-4o",
        })

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(manifest, tmp)
            tmp_path = tmp.name

        try:
            new_agents = provisioner.check_hot_reload(tmp_path)
            assert len(new_agents) >= 1
            new_ids = {a.agent_id for a in new_agents}
            assert "dev-new-hot-001" in new_ids
        finally:
            os.unlink(tmp_path)

    def test_hot_reload_removes_deprovisioned(self, provisioner):
        provisioner.provision_from_manifest(TEST_MANIFEST)
        initial_count = len(provisioner.provisioned_agents)

        # Create manifest without custom_agents
        with open(TEST_MANIFEST) as f:
            manifest = yaml.safe_load(f)

        manifest["agents"]["custom_agents"] = []

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(manifest, tmp)
            tmp_path = tmp.name

        try:
            provisioner.check_hot_reload(tmp_path)
            # Should have fewer agents now
            assert len(provisioner.provisioned_agents) < initial_count
        finally:
            os.unlink(tmp_path)


class TestProvisioningRetry:

    def test_retry_on_failure(self, provisioner_deps):
        """Provisioning retries up to 3 times on failure."""
        rr, ar, factory, bus, qdrant = provisioner_deps

        call_count = {"n": 0}
        original_register = ar.register_agent

        def failing_register(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("Simulated failure")
            return original_register(*args, **kwargs)

        ar.register_agent = failing_register

        # Load roles first
        prov = DynamicProvisioner(rr, ar, factory, bus, qdrant)
        prov.load_role_definitions(yaml.safe_load(open(TEST_MANIFEST)))

        config = AgentConfig(agent_id="retry-001", role_name="pm_agent")
        agent = prov.provision_agent(config)
        assert agent is not None
        assert call_count["n"] == 3  # Failed twice, succeeded on third

    def test_exhausted_retries_raises(self, provisioner_deps):
        """After 3 failed attempts, raises ProvisioningError."""
        rr, ar, factory, bus, qdrant = provisioner_deps

        def always_fail(*args, **kwargs):
            raise RuntimeError("Always fails")

        ar.register_agent = always_fail

        prov = DynamicProvisioner(rr, ar, factory, bus, qdrant)
        prov.load_role_definitions(yaml.safe_load(open(TEST_MANIFEST)))

        config = AgentConfig(agent_id="fail-001", role_name="pm_agent")
        with pytest.raises(ProvisioningError, match="Failed to provision"):
            prov.provision_agent(config)
