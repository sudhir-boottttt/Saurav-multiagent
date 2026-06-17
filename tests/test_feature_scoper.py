"""Property tests and unit tests for Feature Scoping.

Property tests cover Property 25 from the design document.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.feature_scoper import FeatureBreakdown, FeatureScoper, Story
from tests.generators import uuid_st, non_empty_str, short_str


# ──────────────────────────────────────────────
# Property 25: Feature Decomposition Invariant
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 25: Feature Decomposition Invariant


story_st = st.builds(
    dict,
    title=non_empty_str,
    repo=short_str,
    assigned_agent_id=uuid_st,
    acceptance_criteria=st.lists(non_empty_str, min_size=1, max_size=3),
    affected_files=st.lists(st.builds(lambda n: f"src/{n}.py", n=short_str), min_size=0, max_size=3),
    complexity=st.sampled_from(["low", "medium", "high"]),
    depends_on=st.just([]),
)


@given(
    feature_title=non_empty_str,
    tech_lead_id=uuid_st,
    stories=st.lists(story_st, min_size=1, max_size=5),
)
@settings(max_examples=100)
def test_property_25_feature_decomposition(feature_title, tech_lead_id, stories):
    """Each story SHALL be scoped to exactly one repo, assigned to a dev agent,
    and the breakdown SHALL not be sent to DevOps until PM approves.
    """
    scoper = FeatureScoper()
    breakdown = scoper.create_breakdown(feature_title, tech_lead_id, stories)

    # Each story scoped to one repo
    for s in breakdown.stories:
        assert s.repo, "Each story must be scoped to a repo"

    # Not sent to DevOps before approval
    assert not breakdown.sent_to_devops
    with pytest.raises(ValueError, match="approved by PM"):
        scoper.send_to_devops(breakdown.breakdown_id)

    # After approval, can send
    scoper.approve_breakdown(breakdown.breakdown_id)
    scoper.send_to_devops(breakdown.breakdown_id)
    assert breakdown.sent_to_devops


class TestFeatureScoper:

    def test_create_breakdown(self):
        scoper = FeatureScoper()
        stories = [
            {"title": "Add endpoint", "repo": "receiving-svc", "assigned_agent_id": "dev-001"},
            {"title": "Update UI", "repo": "web-app", "assigned_agent_id": "dev-002"},
        ]
        bd = scoper.create_breakdown("New feature", "tl-001", stories)
        assert len(bd.stories) == 2
        assert bd.feature_title == "New feature"

    def test_story_without_repo_raises(self):
        scoper = FeatureScoper()
        with pytest.raises(ValueError, match="scoped to exactly one repo"):
            scoper.create_breakdown("feat", "tl-001", [{"title": "no repo"}])

    def test_approval_flow(self):
        scoper = FeatureScoper()
        bd = scoper.create_breakdown("feat", "tl-001", [
            {"title": "s1", "repo": "r1", "assigned_agent_id": "d1"},
        ])
        assert not bd.approved_by_pm
        scoper.approve_breakdown(bd.breakdown_id)
        assert bd.approved_by_pm

    def test_send_without_approval_raises(self):
        scoper = FeatureScoper()
        bd = scoper.create_breakdown("feat", "tl-001", [
            {"title": "s1", "repo": "r1", "assigned_agent_id": "d1"},
        ])
        with pytest.raises(ValueError):
            scoper.send_to_devops(bd.breakdown_id)

    def test_execution_order_respects_dependencies(self):
        scoper = FeatureScoper()
        stories = [
            {"title": "s1", "repo": "r1", "assigned_agent_id": "d1", "depends_on": []},
            {"title": "s2", "repo": "r2", "assigned_agent_id": "d2", "depends_on": []},
        ]
        bd = scoper.create_breakdown("feat", "tl-001", stories)
        # Make s2 depend on s1
        bd.stories[1].depends_on = [bd.stories[0].story_id]
        order = scoper.get_execution_order(bd.breakdown_id)
        s1_idx = next(i for i, s in enumerate(order) if s.story_id == bd.stories[0].story_id)
        s2_idx = next(i for i, s in enumerate(order) if s.story_id == bd.stories[1].story_id)
        assert s1_idx < s2_idx

    def test_validate_breakdown_missing_agent(self):
        bd = FeatureBreakdown(
            feature_title="feat", tech_lead_id="tl-001",
            stories=[Story(title="s1", repo="r1", assigned_agent_id="")],
        )
        errors = FeatureScoper.validate_breakdown(bd)
        assert any("not assigned" in e for e in errors)

    def test_get_nonexistent_raises(self):
        scoper = FeatureScoper()
        with pytest.raises(KeyError):
            scoper.get_breakdown("nonexistent")
