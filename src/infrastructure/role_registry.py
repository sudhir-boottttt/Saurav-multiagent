"""Plugin-based Role Registry for dynamic agent role management.

Roles are defined in the YAML manifest and loaded at startup.
New roles (DBA, Staff Engineer, Consultant, etc.) can be added
without code changes — just update the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from src.core.models import RoleDefinition


@dataclass
class SpecializationDef:
    """A developer specialization definition from the manifest."""

    additional_tools: list[str] = field(default_factory=list)
    prompt_focus: str = ""


class RoleRegistry:
    """Central registry for agent role definitions.

    Loaded from YAML manifest. Supports built-in and custom roles.
    """

    def __init__(self) -> None:
        self._roles: dict[str, RoleDefinition] = {}
        self._specializations: dict[str, SpecializationDef] = {}

    # --- Role management ---

    def register_role(self, role_def: RoleDefinition) -> RoleDefinition:
        """Register a role definition. Raises ValueError on validation failure."""
        errors = self.validate_role(role_def)
        if errors:
            raise ValueError(f"Invalid role '{role_def.role_name}': {'; '.join(errors)}")
        self._roles[role_def.role_name] = role_def
        return role_def

    def get_role(self, role_name: str) -> RoleDefinition:
        """Get a role by name. Raises KeyError if not found."""
        if role_name not in self._roles:
            raise KeyError(f"Role '{role_name}' not found in registry")
        return self._roles[role_name]

    def list_roles(self, tier_filter: Optional[str] = None) -> list[RoleDefinition]:
        """List all roles, optionally filtered by tier."""
        roles = list(self._roles.values())
        if tier_filter:
            roles = [r for r in roles if r.tier == tier_filter]
        return roles

    def validate_role(self, role_def: RoleDefinition) -> list[str]:
        """Validate a role definition. Returns list of error messages (empty = valid)."""
        errors: list[str] = []
        if not role_def.role_name:
            errors.append("role_name must be non-empty")
        if not role_def.tier:
            errors.append("tier must be non-empty")
        if not role_def.capabilities:
            errors.append("capabilities must have at least one entry")
        # Validate allowed_parent_roles reference existing roles
        for parent in role_def.allowed_parent_roles:
            if parent not in self._roles and parent != role_def.role_name:
                errors.append(f"allowed_parent_role '{parent}' not found in registry")
        return errors

    def get_tools_for_role(self, role_name: str, specialization: Optional[str] = None) -> list[str]:
        """Get the merged tool list for a role + optional specialization."""
        role = self.get_role(role_name)
        tools = list(role.tools)
        if specialization and specialization in self._specializations:
            spec_def = self._specializations[specialization]
            tools.extend(spec_def.additional_tools)
        return tools

    def get_system_prompt(self, role_name: str, specialization: Optional[str] = None) -> str:
        """Get the system prompt template, with specialization focus appended if applicable."""
        role = self.get_role(role_name)
        prompt = role.system_prompt_template
        if specialization and specialization in self._specializations:
            spec_def = self._specializations[specialization]
            if spec_def.prompt_focus:
                prompt += f"\nSpecialization focus: {spec_def.prompt_focus}"
        return prompt

    def get_allowed_parents(self, role_name: str) -> list[str]:
        """Get the list of allowed parent roles for a given role."""
        role = self.get_role(role_name)
        return list(role.allowed_parent_roles)

    def is_valid_parent(self, child_role: str, parent_role: str) -> bool:
        """Check if parent_role is a valid parent for child_role."""
        allowed = self.get_allowed_parents(child_role)
        if not allowed:
            return True  # Top-level roles have no parent constraint
        return parent_role in allowed

    # --- Specialization management ---

    def register_specialization(self, name: str, spec_def: SpecializationDef) -> None:
        """Register a developer specialization."""
        self._specializations[name] = spec_def

    def get_specialization(self, name: str) -> Optional[SpecializationDef]:
        """Get a specialization definition by name."""
        return self._specializations.get(name)

    # --- YAML manifest loading ---

    def load_from_manifest(self, manifest_path: str) -> None:
        """Load role definitions and specializations from a YAML manifest file."""
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        self.load_from_dict(manifest)

    def load_from_dict(self, manifest: dict[str, Any]) -> None:
        """Load role definitions and specializations from a parsed manifest dict."""
        # Load role definitions
        role_defs = manifest.get("role_definitions", {})
        for role_name, role_data in role_defs.items():
            role_def = RoleDefinition(
                role_name=role_name,
                tier=role_data.get("tier", "bottom"),
                capabilities=role_data.get("capabilities", []),
                tools=role_data.get("tools", []),
                system_prompt_template=role_data.get("system_prompt_template", ""),
                allowed_parent_roles=role_data.get("allowed_parent_roles", []),
                allowed_domains=role_data.get("allowed_domains"),
                is_builtin=role_name in ("pm_agent", "tech_lead_agent", "developer_agent"),
            )
            self._roles[role_name] = role_def

        # Load specializations
        specs = manifest.get("specializations", {})
        for spec_name, spec_data in specs.items():
            self._specializations[spec_name] = SpecializationDef(
                additional_tools=spec_data.get("additional_tools", []),
                prompt_focus=spec_data.get("prompt_focus", ""),
            )
