"""Property-based and unit tests for the Role Registry."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.core.models import RoleDefinition
from src.infrastructure.role_registry import RoleRegistry, SpecializationDef


# --- Fixtures ---


def _make_registry_with_builtins() -> RoleRegistry:
    """Create a registry pre-loaded with built-in roles."""
    reg = RoleRegistry()
    reg.register_role(RoleDefinition(
        role_name="pm_agent", tier="top", capabilities=["coordination"],
        tools=["DevOps.create"], allowed_parent_roles=[], is_builtin=True,
    ))
    reg.register_role(RoleDefinition(
        role_name="tech_lead_agent", tier="middle", capabilities=["review"],
        tools=["CodeIndex.search"], allowed_parent_roles=["pm_agent"], is_builtin=True,
    ))
    reg.register_role(RoleDefinition(
        role_name="developer_agent", tier="bottom", capabilities=["code_analysis"],
        tools=["GitHub.clone"], allowed_parent_roles=["tech_lead_agent"], is_builtin=True,
    ))
    return reg


# --- Property tests ---
# Feature: multi-agent-platform, Property: Role Registration Invariant


@given(
    role_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    tier=st.sampled_from(["top", "middle", "bottom", "cross-domain"]),
    capabilities=st.lists(st.text(min_size=1, max_size=10, alphabet="abcdefghij"), min_size=1, max_size=3),
)
@settings(max_examples=100)
def test_registered_role_is_retrievable(role_name, tier, capabilities):
    """Any successfully registered role must be retrievable by name."""
    reg = RoleRegistry()
    role_def = RoleDefinition(role_name=role_name, tier=tier, capabilities=capabilities)
    reg.register_role(role_def)
    retrieved = reg.get_role(role_name)
    assert retrieved.role_name == role_name
    assert retrieved.tier == tier
    assert retrieved.capabilities == capabilities


# Feature: multi-agent-platform, Property: Parent Validation

@given(
    child_role=st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
    parent_role=st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
)
@settings(max_examples=100)
def test_parent_validation_consistency(child_role, parent_role):
    """is_valid_parent should return True iff parent is in allowed_parent_roles (or allowed is empty)."""
    reg = RoleRegistry()
    # Register parent first
    reg.register_role(RoleDefinition(role_name=parent_role, tier="middle", capabilities=["cap"]))
    # Register child with parent in allowed list
    reg.register_role(RoleDefinition(
        role_name=child_role, tier="bottom", capabilities=["cap"],
        allowed_parent_roles=[parent_role],
    ))
    assert reg.is_valid_parent(child_role, parent_role) is True


# Feature: multi-agent-platform, Property: Specialization Tool Merging

@given(
    base_tools=st.lists(st.text(min_size=1, max_size=10, alphabet="abcdefghij"), min_size=1, max_size=3),
    extra_tools=st.lists(st.text(min_size=1, max_size=10, alphabet="abcdefghij"), min_size=1, max_size=3),
)
@settings(max_examples=100)
def test_specialization_merges_tools(base_tools, extra_tools):
    """Tools from specialization should be appended to base role tools."""
    reg = RoleRegistry()
    reg.register_role(RoleDefinition(
        role_name="dev", tier="bottom", capabilities=["cap"], tools=base_tools,
    ))
    reg.register_specialization("frontend", SpecializationDef(additional_tools=extra_tools))
    merged = reg.get_tools_for_role("dev", specialization="frontend")
    assert merged == base_tools + extra_tools


# --- Unit tests ---


def test_load_builtin_roles():
    reg = _make_registry_with_builtins()
    assert len(reg.list_roles()) == 3
    assert reg.get_role("pm_agent").is_builtin is True
    assert reg.get_role("tech_lead_agent").tier == "middle"
    assert reg.get_role("developer_agent").tier == "bottom"


def test_load_custom_roles():
    reg = _make_registry_with_builtins()
    reg.register_role(RoleDefinition(
        role_name="db_admin", tier="bottom", capabilities=["schema_analysis"],
        tools=["DBMigration.scan"], allowed_parent_roles=["tech_lead_agent"],
    ))
    reg.register_role(RoleDefinition(
        role_name="staff_engineer", tier="cross-domain",
        capabilities=["architecture_review"],
        tools=["ArchAnalyzer.review"], allowed_parent_roles=["pm_agent"],
    ))
    reg.register_role(RoleDefinition(
        role_name="consultant", tier="cross-domain",
        capabilities=["best_practice_audit"],
        tools=["CodeIndex.search"], allowed_parent_roles=["pm_agent", "tech_lead_agent"],
    ))
    assert len(reg.list_roles()) == 6
    assert reg.get_role("db_admin").tier == "bottom"
    assert reg.get_role("staff_engineer").tier == "cross-domain"
    assert reg.is_valid_parent("consultant", "pm_agent") is True
    assert reg.is_valid_parent("consultant", "tech_lead_agent") is True


def test_invalid_role_rejected_empty_capabilities():
    reg = RoleRegistry()
    with pytest.raises(ValueError, match="capabilities"):
        reg.register_role(RoleDefinition(role_name="bad", tier="bottom", capabilities=[]))


def test_get_nonexistent_role_raises():
    reg = RoleRegistry()
    with pytest.raises(KeyError, match="not found"):
        reg.get_role("nonexistent")


def test_list_roles_with_tier_filter():
    reg = _make_registry_with_builtins()
    middle = reg.list_roles(tier_filter="middle")
    assert len(middle) == 1
    assert middle[0].role_name == "tech_lead_agent"


def test_system_prompt_with_specialization():
    reg = RoleRegistry()
    reg.register_role(RoleDefinition(
        role_name="dev", tier="bottom", capabilities=["cap"],
        system_prompt_template="You are a developer.",
    ))
    reg.register_specialization("frontend", SpecializationDef(
        prompt_focus="UI components, browser APIs",
    ))
    prompt = reg.get_system_prompt("dev", specialization="frontend")
    assert "You are a developer." in prompt
    assert "UI components, browser APIs" in prompt


def test_load_from_dict():
    manifest = {
        "role_definitions": {
            "pm_agent": {
                "tier": "top",
                "capabilities": ["coordination"],
                "tools": ["DevOps.create"],
                "allowed_parent_roles": [],
            },
            "custom_role": {
                "tier": "bottom",
                "capabilities": ["custom_cap"],
                "tools": ["Custom.tool"],
                "allowed_parent_roles": ["pm_agent"],
            },
        },
        "specializations": {
            "frontend": {
                "additional_tools": ["Lint.eslint"],
                "prompt_focus": "UI focus",
            },
        },
    }
    reg = RoleRegistry()
    reg.load_from_dict(manifest)
    assert len(reg.list_roles()) == 2
    assert reg.get_role("pm_agent").is_builtin is True
    assert reg.get_role("custom_role").is_builtin is False
    assert reg.get_specialization("frontend") is not None
    assert reg.get_specialization("frontend").prompt_focus == "UI focus"


def test_is_valid_parent_rejects_invalid():
    reg = _make_registry_with_builtins()
    # developer_agent only allows tech_lead_agent as parent
    assert reg.is_valid_parent("developer_agent", "pm_agent") is False
    assert reg.is_valid_parent("developer_agent", "tech_lead_agent") is True
