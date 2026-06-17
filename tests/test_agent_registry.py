"""Property-based and unit tests for the Agent Registry."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.core.enums import AgentStatus
from src.core.models import AgentRecord
from src.infrastructure.agent_registry import AgentFilter, AgentRegistry


# --- Property Tests ---
# Feature: multi-agent-platform, Property 1: Agent Registration Invariant


@given(
    role=st.sampled_from(["pm_agent", "tech_lead_agent", "developer_agent", "db_admin"]),
    specialization=st.one_of(st.none(), st.sampled_from(["frontend", "backend", "fullstack"])),
)
@settings(max_examples=100)
def test_registered_agent_is_retrievable(role, specialization):
    """Any registered agent must be retrievable with all submitted fields."""
    reg = AgentRegistry()
    agent_id = str(uuid4())
    agent = AgentRecord(
        agent_id=agent_id,
        role=role,
        specialization=specialization,
        capabilities=["cap1"],
    )
    reg.register_agent(agent)
    retrieved = reg.get_agent(agent_id)
    assert retrieved.agent_id == agent_id
    assert retrieved.role == role
    assert retrieved.specialization == specialization
    assert retrieved.status == AgentStatus.STARTING


# Feature: multi-agent-platform, Property 2: Agent Health Lifecycle


def test_health_lifecycle_unhealthy_after_30s():
    """Agent should be marked unhealthy after 30s without heartbeat."""
    reg = AgentRegistry()
    agent = AgentRecord(agent_id=str(uuid4()), role="developer_agent", capabilities=["cap"])
    reg.register_agent(agent)
    # Simulate 31 seconds without heartbeat
    agent.last_heartbeat = datetime.now(UTC) - timedelta(seconds=31)
    agent.status = AgentStatus.HEALTHY
    results = reg.health_check()
    assert results[agent.agent_id] == AgentStatus.UNHEALTHY


def test_health_lifecycle_needs_restart_after_120s():
    """Agent should need restart after 120s without heartbeat."""
    reg = AgentRegistry()
    agent = AgentRecord(agent_id=str(uuid4()), role="developer_agent", capabilities=["cap"])
    reg.register_agent(agent)
    agent.last_heartbeat = datetime.now(UTC) - timedelta(seconds=121)
    agent.status = AgentStatus.HEALTHY
    restart_ids = reg.get_agents_needing_restart()
    assert agent.agent_id in restart_ids


def test_healthy_agent_not_flagged():
    """Agent with recent heartbeat should remain healthy."""
    reg = AgentRegistry()
    agent = AgentRecord(agent_id=str(uuid4()), role="developer_agent", capabilities=["cap"])
    reg.register_agent(agent)
    reg.record_heartbeat(agent.agent_id)
    results = reg.health_check()
    assert results[agent.agent_id] == AgentStatus.HEALTHY


# Feature: multi-agent-platform, Property 3: Agent Hierarchy Invariant


def test_hierarchy_pm_tech_lead_developer():
    """Platform should support the three-tier hierarchy."""
    reg = AgentRegistry()
    pm = AgentRecord(agent_id=str(uuid4()), role="pm_agent", capabilities=["coord"])
    tl = AgentRecord(
        agent_id=str(uuid4()), role="tech_lead_agent",
        domain=["inbound"], parent_agent_id=pm.agent_id, capabilities=["review"],
    )
    dev = AgentRecord(
        agent_id=str(uuid4()), role="developer_agent",
        domain=["inbound"], parent_agent_id=tl.agent_id,
        assigned_repos=["receiving-svc"], capabilities=["code"],
    )
    reg.register_agent(pm)
    reg.register_agent(tl)
    reg.register_agent(dev)

    assert len(reg.list_agents()) == 3
    assert reg.get_agent(dev.agent_id).parent_agent_id == tl.agent_id
    assert reg.get_agent(tl.agent_id).parent_agent_id == pm.agent_id


# --- Unit Tests ---


def test_duplicate_registration_raises():
    reg = AgentRegistry()
    agent_id = str(uuid4())
    agent = AgentRecord(agent_id=agent_id, role="pm_agent", capabilities=["cap"])
    reg.register_agent(agent)
    with pytest.raises(ValueError, match="already registered"):
        reg.register_agent(AgentRecord(agent_id=agent_id, role="pm_agent", capabilities=["cap"]))


def test_deregister_agent():
    reg = AgentRegistry()
    agent = AgentRecord(agent_id=str(uuid4()), role="pm_agent", capabilities=["cap"])
    reg.register_agent(agent)
    assert reg.deregister_agent(agent.agent_id) is True
    assert reg.deregister_agent(agent.agent_id) is False  # already removed


def test_get_nonexistent_agent_raises():
    reg = AgentRegistry()
    with pytest.raises(KeyError, match="not found"):
        reg.get_agent("nonexistent")


def test_list_agents_with_filter():
    reg = AgentRegistry()
    a1 = AgentRecord(agent_id=str(uuid4()), role="developer_agent", domain=["inbound"], capabilities=["cap"])
    a2 = AgentRecord(agent_id=str(uuid4()), role="developer_agent", domain=["outbound"], capabilities=["cap"])
    a3 = AgentRecord(agent_id=str(uuid4()), role="tech_lead_agent", domain=["inbound"], capabilities=["cap"])
    reg.register_agent(a1)
    reg.register_agent(a2)
    reg.register_agent(a3)

    devs = reg.list_agents(AgentFilter(role="developer_agent"))
    assert len(devs) == 2

    inbound = reg.list_agents(AgentFilter(domain="inbound"))
    assert len(inbound) == 2

    inbound_devs = reg.list_agents(AgentFilter(role="developer_agent", domain="inbound"))
    assert len(inbound_devs) == 1


def test_update_status():
    reg = AgentRegistry()
    agent = AgentRecord(agent_id=str(uuid4()), role="pm_agent", capabilities=["cap"])
    reg.register_agent(agent)
    reg.update_status(agent.agent_id, AgentStatus.HEALTHY)
    assert reg.get_agent(agent.agent_id).status == AgentStatus.HEALTHY
    reg.update_status(agent.agent_id, AgentStatus.STOPPED)
    assert reg.get_agent(agent.agent_id).status == AgentStatus.STOPPED


def test_record_heartbeat_marks_healthy():
    reg = AgentRegistry()
    agent = AgentRecord(agent_id=str(uuid4()), role="pm_agent", capabilities=["cap"])
    reg.register_agent(agent)
    assert agent.status == AgentStatus.STARTING
    reg.record_heartbeat(agent.agent_id)
    assert reg.get_agent(agent.agent_id).status == AgentStatus.HEALTHY


def test_stopped_agent_not_flagged_for_restart():
    reg = AgentRegistry()
    agent = AgentRecord(agent_id=str(uuid4()), role="pm_agent", capabilities=["cap"])
    reg.register_agent(agent)
    reg.update_status(agent.agent_id, AgentStatus.STOPPED)
    agent.last_heartbeat = datetime.now(UTC) - timedelta(seconds=200)
    assert agent.agent_id not in reg.get_agents_needing_restart()
