"""Feature scoping and story breakdown workflow.

Handles PM → Tech Lead feature assignment, story decomposition
scoped to single repos, dependency ordering, and PM approval
before DevOps work item creation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Story:
    """A user story scoped to a single repository."""
    story_id: str = ""
    title: str = ""
    repo: str = ""
    assigned_agent_id: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    complexity: str = "medium"  # low | medium | high
    depends_on: list[str] = field(default_factory=list)  # story_ids
    order: int = 0

    def __post_init__(self):
        if not self.story_id:
            self.story_id = str(uuid4())


@dataclass
class FeatureBreakdown:
    """A complete feature breakdown with stories and approval status."""
    breakdown_id: str = ""
    feature_title: str = ""
    tech_lead_id: str = ""
    stories: list[Story] = field(default_factory=list)
    approved_by_pm: bool = False
    sent_to_devops: bool = False

    def __post_init__(self):
        if not self.breakdown_id:
            self.breakdown_id = str(uuid4())


class FeatureScoper:
    """Manages feature scoping and story breakdown workflow."""

    def __init__(self) -> None:
        self._breakdowns: dict[str, FeatureBreakdown] = {}

    def create_breakdown(
        self,
        feature_title: str,
        tech_lead_id: str,
        stories: list[dict[str, Any]],
    ) -> FeatureBreakdown:
        """Create a feature breakdown with stories.

        Each story must be scoped to exactly one repo (Property 25).
        """
        parsed_stories = []
        for i, s in enumerate(stories):
            if not s.get("repo"):
                raise ValueError(f"Story {i} must be scoped to exactly one repo")
            story = Story(
                title=s.get("title", ""),
                repo=s["repo"],
                assigned_agent_id=s.get("assigned_agent_id", ""),
                acceptance_criteria=s.get("acceptance_criteria", []),
                affected_files=s.get("affected_files", []),
                complexity=s.get("complexity", "medium"),
                depends_on=s.get("depends_on", []),
                order=i,
            )
            parsed_stories.append(story)

        breakdown = FeatureBreakdown(
            feature_title=feature_title,
            tech_lead_id=tech_lead_id,
            stories=parsed_stories,
        )
        self._breakdowns[breakdown.breakdown_id] = breakdown
        logger.info("Created breakdown %s with %d stories", breakdown.breakdown_id, len(parsed_stories))
        return breakdown

    def get_breakdown(self, breakdown_id: str) -> FeatureBreakdown:
        if breakdown_id not in self._breakdowns:
            raise KeyError(f"Breakdown {breakdown_id} not found")
        return self._breakdowns[breakdown_id]

    def approve_breakdown(self, breakdown_id: str) -> FeatureBreakdown:
        """PM approves the breakdown. Must be approved before sending to DevOps."""
        breakdown = self.get_breakdown(breakdown_id)
        breakdown.approved_by_pm = True
        logger.info("Breakdown %s approved by PM", breakdown_id)
        return breakdown

    def send_to_devops(self, breakdown_id: str) -> FeatureBreakdown:
        """Send breakdown to DevOps for work item creation.

        Raises ValueError if not yet approved by PM (Property 25).
        """
        breakdown = self.get_breakdown(breakdown_id)
        if not breakdown.approved_by_pm:
            raise ValueError("Breakdown must be approved by PM before sending to DevOps")
        breakdown.sent_to_devops = True
        logger.info("Breakdown %s sent to DevOps", breakdown_id)
        return breakdown

    def get_execution_order(self, breakdown_id: str) -> list[Story]:
        """Get stories in dependency-respecting execution order."""
        breakdown = self.get_breakdown(breakdown_id)
        # Topological sort by depends_on
        stories_by_id = {s.story_id: s for s in breakdown.stories}
        visited = set()
        order = []

        def visit(story_id: str):
            if story_id in visited:
                return
            visited.add(story_id)
            story = stories_by_id.get(story_id)
            if story:
                for dep in story.depends_on:
                    visit(dep)
                order.append(story)

        for s in breakdown.stories:
            visit(s.story_id)
        return order

    @staticmethod
    def validate_breakdown(breakdown: FeatureBreakdown) -> list[str]:
        """Validate breakdown per Property 25. Returns list of errors."""
        errors = []
        for s in breakdown.stories:
            if not s.repo:
                errors.append(f"Story {s.story_id} not scoped to a repo")
            if not s.assigned_agent_id:
                errors.append(f"Story {s.story_id} not assigned to an agent")
        if breakdown.sent_to_devops and not breakdown.approved_by_pm:
            errors.append("Breakdown sent to DevOps without PM approval")
        return errors
