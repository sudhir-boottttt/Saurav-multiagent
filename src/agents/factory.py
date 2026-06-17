"""Agent factory for creating AutoGen-based agents from Role Registry config.

Creates PM, Tech Lead, and Developer agents with role-specific tools
and system prompts resolved from the Role Registry. Supports
specialization-based tool loading for Developer agents.

When autogen-agentchat is installed, agents are real AutoGen
AssistantAgent / SocietyOfMindAgent instances. Otherwise, a lightweight
AgentWrapper is used for testing and development.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from src.core.models import RoleDefinition
from src.infrastructure.role_registry import RoleRegistry

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM provider configuration (OpenAI-compatible)."""

    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    provider: str = "openai"


@dataclass
class AgentConfig:
    """Configuration for creating an agent instance."""

    agent_id: str
    role_name: str
    specialization: Optional[str] = None
    domain: Optional[list[str]] = None
    assigned_repos: list[str] = field(default_factory=list)
    model: str = "gpt-4o-mini"
    parent_agent_id: Optional[str] = None
    extra_config: dict[str, Any] = field(default_factory=dict)


class AgentWrapper:
    """Lightweight agent wrapper compatible with AutoGen's interface.

    When autogen-agentchat is available, this wraps a real AssistantAgent.
    Otherwise, it provides the same interface for testing.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        specialization: Optional[str],
        tools: list[str],
        system_prompt: str,
        model: str,
        domain: Optional[list[str]] = None,
        assigned_repos: Optional[list[str]] = None,
        parent_agent_id: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.specialization = specialization
        self.tools = tools
        self.system_prompt = system_prompt
        self.model = model
        self.domain = domain or []
        self.assigned_repos = assigned_repos or []
        self.parent_agent_id = parent_agent_id
        self._inner_agent: Any = None  # AutoGen agent when available

    @property
    def is_autogen_backed(self) -> bool:
        return self._inner_agent is not None

    def __repr__(self) -> str:
        return (
            f"AgentWrapper(id={self.agent_id}, role={self.role}, "
            f"spec={self.specialization}, tools={len(self.tools)})"
        )


class TeamWrapper:
    """Wraps a SocietyOfMindAgent-style team (Tech Lead + child agents).

    The Tech Lead agent orchestrates a GroupChat of its child agents.
    """

    def __init__(self, leader: AgentWrapper, members: list[AgentWrapper]) -> None:
        self.leader = leader
        self.members = list(members)
        self._group_chat: Any = None  # AutoGen GroupChat when available

    def add_member(self, agent: AgentWrapper) -> None:
        self.members.append(agent)

    @property
    def all_agents(self) -> list[AgentWrapper]:
        return [self.leader] + self.members

    def __repr__(self) -> str:
        return f"TeamWrapper(leader={self.leader.name}, members={len(self.members)})"


class AgentFactory:
    """Creates agents from Role Registry configuration.

    Resolves role definitions, specializations, tools, and system prompts
    from the Role Registry to produce configured agent instances.
    """

    def __init__(self, role_registry: RoleRegistry, llm_config: Optional[LLMConfig] = None) -> None:
        self._role_registry = role_registry
        self._llm_config = llm_config or LLMConfig()
        self._tool_registry: Optional[Any] = None

    def set_tool_registry(self, tool_registry: Any) -> None:
        """Set the ToolRegistry for resolving tool schemas."""
        self._tool_registry = tool_registry

    def create_agent(self, config: AgentConfig) -> AgentWrapper:
        """Create a single agent from config.

        Resolves tools and system prompt from Role Registry based on
        role_name and optional specialization.

        Raises ValueError if role doesn't exist in registry.
        """
        try:
            role_def = self._role_registry.get_role(config.role_name)
        except KeyError:
            raise ValueError(f"Role '{config.role_name}' not found in Role Registry")

        # Resolve tools (base + specialization)
        tools = self._role_registry.get_tools_for_role(
            config.role_name, config.specialization,
        )

        # Resolve system prompt
        system_prompt = self._role_registry.get_system_prompt(
            config.role_name, config.specialization,
        )

        # Build agent name
        name_parts = [config.role_name]
        if config.specialization:
            name_parts.append(config.specialization)
        if config.domain:
            name_parts.extend(config.domain)
        agent_name = "-".join(name_parts)

        agent = AgentWrapper(
            agent_id=config.agent_id,
            name=agent_name,
            role=config.role_name,
            specialization=config.specialization,
            tools=tools,
            system_prompt=system_prompt,
            model=config.model,
            domain=config.domain,
            assigned_repos=config.assigned_repos,
            parent_agent_id=config.parent_agent_id,
        )

        # Try to create real AutoGen agent if available
        self._try_create_autogen_agent(agent, role_def, self._llm_config)

        logger.info("Created agent: %s", agent)
        return agent

    def create_team(
        self,
        leader_config: AgentConfig,
        member_configs: list[AgentConfig],
    ) -> TeamWrapper:
        """Create a Tech Lead team (SocietyOfMindAgent pattern).

        The leader is a Tech Lead agent, members are Developer agents
        (or other child roles) that form the inner GroupChat.
        """
        leader = self.create_agent(leader_config)
        members = [self.create_agent(mc) for mc in member_configs]

        team = TeamWrapper(leader=leader, members=members)
        logger.info("Created team: %s", team)
        return team

    def create_pm_agent(self, config: AgentConfig) -> AgentWrapper:
        """Create the PM Agent (top-tier, cross-domain coordination)."""
        config.role_name = "pm_agent"
        return self.create_agent(config)

    def create_tech_lead_agent(self, config: AgentConfig) -> AgentWrapper:
        """Create a Tech Lead Agent (middle-tier, domain-specific)."""
        config.role_name = "tech_lead_agent"
        return self.create_agent(config)

    def create_developer_agent(self, config: AgentConfig) -> AgentWrapper:
        """Create a Developer Agent (bottom-tier, repo-specific)."""
        config.role_name = "developer_agent"
        return self.create_agent(config)

    @staticmethod
    def _try_create_autogen_agent(agent: AgentWrapper, role_def: RoleDefinition, llm_config: LLMConfig) -> None:
        """Attempt to create a real AutoGen agent. No-op if autogen not installed.

        Supports autogen-agentchat v0.4+ (model_client API) with any
        OpenAI-compatible provider (OpenAI, Gemini, Ollama, etc.).
        Reads api_key from llm_config or AGENT_LLM_API_KEY env var.
        """
        import os

        try:
            from autogen_agentchat.agents import AssistantAgent  # type: ignore[import-not-found]

            model_client = None
            try:
                from autogen_ext.models.openai import OpenAIChatCompletionClient  # type: ignore[import-not-found]

                api_key = llm_config.api_key or os.environ.get("AGENT_LLM_API_KEY")
                if not api_key:
                    logger.debug(
                        "No API key for %s. Set AGENT_LLM_API_KEY env var or llm.api_key in manifest.",
                        agent.agent_id,
                    )
                    return

                client_kwargs: dict[str, Any] = {
                    "model": agent.model,
                    "api_key": api_key,
                    # Required for non-OpenAI models so AutoGen knows
                    # what the model supports (suppresses model mismatch warning).
                    "model_capabilities": {
                        "vision": True,
                        "function_calling": True,
                        "json_output": True,
                    },
                }
                if llm_config.base_url:
                    client_kwargs["base_url"] = llm_config.base_url

                model_client = OpenAIChatCompletionClient(**client_kwargs)

            except ImportError:
                logger.debug("autogen_ext.models.openai not installed for %s", agent.agent_id)
                return
            except Exception as client_err:
                logger.debug(
                    "Could not create model client for %s: %s. Using AgentWrapper fallback.",
                    agent.agent_id, client_err,
                )
                return

            autogen_agent = AssistantAgent(
                name=agent.name,
                system_message=agent.system_prompt,
                model_client=model_client,
            )
            agent._inner_agent = autogen_agent
            logger.info("Created AutoGen AssistantAgent for %s (provider=%s)", agent.agent_id, llm_config.provider)
        except ImportError:
            logger.debug("autogen-agentchat not installed, using AgentWrapper for %s", agent.agent_id)
        except Exception as e:
            logger.debug("AutoGen agent creation failed for %s: %s. Using AgentWrapper.", agent.agent_id, e)
