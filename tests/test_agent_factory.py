"""Unit tests for the Agent Factory.

Tests agent creation with various role/specialization combinations.
"""

import pytest

from src.agents.factory import AgentConfig, AgentFactory, AgentWrapper, TeamWrapper
from src.core.models import RoleDefinition
from src.infrastructure.role_registry import RoleRegistry, SpecializationDef


@pytest.fixture
def role_registry():
    """Create a Role Registry with built-in and custom roles."""
    rr = RoleRegistry()
    rr.register_role(RoleDefinition(
        role_name="pm_agent", tier="top",
        capabilities=["feature_prioritization", "cross_domain_coordination"],
        tools=["DevOpsConnector.create_work_item", "MessageBus.publish"],
        system_prompt_template="You are the PM Agent. {{ domain_context }}",
        allowed_parent_roles=[], is_builtin=True,
    ))
    rr.register_role(RoleDefinition(
        role_name="tech_lead_agent", tier="middle",
        capabilities=["feature_breakdown", "code_review"],
        tools=["CodeIndex.search", "DiscussionThread.create"],
        system_prompt_template="You are a Tech Lead for {{ domain }}.",
        allowed_parent_roles=["pm_agent"], is_builtin=True,
    ))
    rr.register_role(RoleDefinition(
        role_name="developer_agent", tier="bottom",
        capabilities=["code_analysis", "bug_detection", "pr_creation"],
        tools=["GitHubConnector.clone_repo", "CodeIndex.search"],
        system_prompt_template="You are a Developer Agent for {{ repo_name }}. Specialization: {{ specialization }}.",
        allowed_parent_roles=["tech_lead_agent"], is_builtin=True,
    ))
    rr.register_role(RoleDefinition(
        role_name="staff_engineer", tier="cross-domain",
        capabilities=["architecture_review", "cross_domain_analysis"],
        tools=["CodeIndex.search", "ArchitectureAnalyzer.review"],
        system_prompt_template="You are a Staff Engineer spanning multiple domains.",
        allowed_parent_roles=["pm_agent"],
    ))
    rr.register_role(RoleDefinition(
        role_name="db_admin", tier="bottom",
        capabilities=["schema_analysis", "migration_generation"],
        tools=["DBMigrationChecker.scan", "SchemaAnalyzer.analyze"],
        system_prompt_template="You are a DB Admin Agent.",
        allowed_parent_roles=["tech_lead_agent"],
    ))

    # Register specializations
    rr.register_specialization("frontend", SpecializationDef(
        additional_tools=["LintRunner.eslint", "ComponentAnalyzer.scan"],
        prompt_focus="UI components, accessibility",
    ))
    rr.register_specialization("backend", SpecializationDef(
        additional_tools=["APIContractValidator.check", "DBMigrationChecker.scan"],
        prompt_focus="API design, data models, security",
    ))
    rr.register_specialization("devops", SpecializationDef(
        additional_tools=["DockerfileAnalyzer.scan", "PipelineValidator.check"],
        prompt_focus="CI/CD, infrastructure",
    ))
    return rr


@pytest.fixture
def factory(role_registry):
    return AgentFactory(role_registry)


class TestAgentFactory:

    def test_create_pm_agent(self, factory):
        config = AgentConfig(agent_id="pm-001", role_name="pm_agent")
        agent = factory.create_pm_agent(config)
        assert agent.agent_id == "pm-001"
        assert agent.role == "pm_agent"
        assert "DevOpsConnector.create_work_item" in agent.tools
        assert agent.system_prompt  # non-empty

    def test_create_tech_lead_agent(self, factory):
        config = AgentConfig(
            agent_id="tl-001", role_name="tech_lead_agent",
            domain=["inbound"],
        )
        agent = factory.create_tech_lead_agent(config)
        assert agent.role == "tech_lead_agent"
        assert agent.domain == ["inbound"]
        assert "CodeIndex.search" in agent.tools

    def test_create_developer_backend(self, factory):
        config = AgentConfig(
            agent_id="dev-001", role_name="developer_agent",
            specialization="backend", domain=["inbound"],
            assigned_repos=["receiving-service"],
        )
        agent = factory.create_developer_agent(config)
        assert agent.role == "developer_agent"
        assert agent.specialization == "backend"
        # Should have base tools + backend specialization tools
        assert "GitHubConnector.clone_repo" in agent.tools
        assert "APIContractValidator.check" in agent.tools

    def test_create_developer_frontend(self, factory):
        config = AgentConfig(
            agent_id="dev-002", role_name="developer_agent",
            specialization="frontend", domain=["inbound"],
            assigned_repos=["web-app"],
        )
        agent = factory.create_developer_agent(config)
        assert "LintRunner.eslint" in agent.tools
        assert "ComponentAnalyzer.scan" in agent.tools

    def test_create_developer_devops(self, factory):
        config = AgentConfig(
            agent_id="dev-003", role_name="developer_agent",
            specialization="devops", domain=["outbound"],
        )
        agent = factory.create_developer_agent(config)
        assert "DockerfileAnalyzer.scan" in agent.tools

    def test_create_cross_domain_staff_engineer(self, factory):
        config = AgentConfig(
            agent_id="se-001", role_name="staff_engineer",
            domain=["inbound", "outbound"],
        )
        agent = factory.create_agent(config)
        assert agent.role == "staff_engineer"
        assert agent.domain == ["inbound", "outbound"]
        assert "ArchitectureAnalyzer.review" in agent.tools

    def test_create_custom_role_db_admin(self, factory):
        config = AgentConfig(
            agent_id="dba-001", role_name="db_admin",
            domain=["inbound"],
        )
        agent = factory.create_agent(config)
        assert agent.role == "db_admin"
        assert "SchemaAnalyzer.analyze" in agent.tools

    def test_create_team(self, factory):
        leader_config = AgentConfig(
            agent_id="tl-001", role_name="tech_lead_agent",
            domain=["inbound"],
        )
        member_configs = [
            AgentConfig(
                agent_id="dev-001", role_name="developer_agent",
                specialization="backend", domain=["inbound"],
                assigned_repos=["receiving-service"],
                parent_agent_id="tl-001",
            ),
            AgentConfig(
                agent_id="dev-002", role_name="developer_agent",
                specialization="backend", domain=["inbound"],
                assigned_repos=["putaway-service"],
                parent_agent_id="tl-001",
            ),
        ]
        team = factory.create_team(leader_config, member_configs)
        assert isinstance(team, TeamWrapper)
        assert team.leader.role == "tech_lead_agent"
        assert len(team.members) == 2
        assert len(team.all_agents) == 3

    def test_nonexistent_role_raises(self, factory):
        config = AgentConfig(agent_id="x-001", role_name="nonexistent_role")
        with pytest.raises(ValueError, match="not found"):
            factory.create_agent(config)

    def test_agent_name_includes_role_and_specialization(self, factory):
        config = AgentConfig(
            agent_id="dev-001", role_name="developer_agent",
            specialization="frontend", domain=["inbound"],
        )
        agent = factory.create_developer_agent(config)
        assert "developer_agent" in agent.name
        assert "frontend" in agent.name

    def test_agent_wrapper_repr(self, factory):
        config = AgentConfig(agent_id="dev-001", role_name="developer_agent")
        agent = factory.create_developer_agent(config)
        r = repr(agent)
        assert "dev-001" in r
        assert "developer_agent" in r

    def test_team_add_member(self, factory):
        leader_config = AgentConfig(agent_id="tl-001", role_name="tech_lead_agent")
        team = factory.create_team(leader_config, [])
        assert len(team.members) == 0

        new_dev = factory.create_agent(AgentConfig(
            agent_id="dev-new", role_name="developer_agent",
            specialization="backend",
        ))
        team.add_member(new_dev)
        assert len(team.members) == 1

    def test_no_autogen_uses_wrapper(self, factory):
        config = AgentConfig(agent_id="dev-001", role_name="developer_agent")
        agent = factory.create_developer_agent(config)
        # Without autogen installed, should use wrapper
        assert not agent.is_autogen_backed
