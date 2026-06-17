"""Property tests and unit tests for Cross-Domain Impact Analysis.

Property tests cover Properties 22-24 from the design document.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.impact_analyzer import ImpactAnalyzer
from src.core.models import ImpactReport
from tests.generators import uuid_st, non_empty_str, short_str


# ──────────────────────────────────────────────
# Property 22: Cross-Domain Impact Trigger
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 22: Cross-Domain Impact Trigger


@given(
    shared_file=st.sampled_from([
        "shared/models.py", "contracts/api.proto", "proto/events.proto", "api/v1/schema.json",
    ]),
    normal_file=st.builds(lambda n: f"src/{n}.py", n=short_str),
)
@settings(max_examples=100)
def test_property_22_cross_domain_trigger(shared_file, normal_file):
    """Changes to shared API/data model files SHALL trigger cross-domain analysis."""
    analyzer = ImpactAnalyzer()
    assert analyzer.detect_cross_domain_change([shared_file])
    # Normal files should not trigger
    assert not analyzer.detect_cross_domain_change([normal_file])


# ──────────────────────────────────────────────
# Property 23: Impact Report Completeness
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 23: Impact Report Completeness


@given(
    trigger_id=uuid_st,
    description=non_empty_str,
    repos=st.lists(short_str, min_size=1, max_size=3),
    endpoints=st.lists(non_empty_str, min_size=1, max_size=3),
    effort=st.sampled_from(["low", "medium", "high"]),
    risk=st.sampled_from(["low", "medium", "high", "critical"]),
    breaking=st.booleans(),
)
@settings(max_examples=100)
def test_property_23_report_completeness(trigger_id, description, repos, endpoints, effort, risk, breaking):
    """Impact reports SHALL contain affected_repos, affected_endpoints,
    estimated_effort, and risk_level (all non-empty).
    """
    analyzer = ImpactAnalyzer()
    report = analyzer.compile_impact_report(
        trigger_agent_id=trigger_id,
        change_description=description,
        affected_repos=repos,
        affected_endpoints=endpoints,
        estimated_effort=effort,
        risk_level=risk,
        has_breaking_changes=breaking,
    )
    assert ImpactAnalyzer.validate_report(report)
    assert len(report.affected_repos) >= 1
    assert len(report.affected_endpoints) >= 1
    assert report.estimated_effort
    assert report.risk_level


# ──────────────────────────────────────────────
# Property 24: Migration Plan Work Item Creation
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 24: Migration Plan Work Item Creation


@given(
    trigger_id=uuid_st,
    breaking=st.just(True),
)
@settings(max_examples=50)
def test_property_24_breaking_change_triggers_discussion(trigger_id, breaking):
    """Breaking changes SHALL trigger a discussion thread (indicated by has_breaking_changes)."""
    analyzer = ImpactAnalyzer()
    report = analyzer.compile_impact_report(
        trigger_agent_id=trigger_id,
        change_description="Breaking API change",
        affected_repos=["svc-a", "svc-b"],
        affected_endpoints=["/api/v1/items"],
        estimated_effort="high",
        risk_level="critical",
        has_breaking_changes=breaking,
    )
    assert report.has_breaking_changes is True


class TestImpactAnalyzer:

    def test_detect_shared_file(self):
        analyzer = ImpactAnalyzer()
        assert analyzer.detect_cross_domain_change(["shared/models.py"])

    def test_no_shared_files(self):
        analyzer = ImpactAnalyzer()
        assert not analyzer.detect_cross_domain_change(["src/main.py", "tests/test.py"])

    def test_empty_repos_raises(self):
        analyzer = ImpactAnalyzer()
        with pytest.raises(ValueError, match="affected_repos"):
            analyzer.compile_impact_report("a", "desc", [], ["/api"], "low", "low")

    def test_empty_endpoints_raises(self):
        analyzer = ImpactAnalyzer()
        with pytest.raises(ValueError, match="affected_endpoints"):
            analyzer.compile_impact_report("a", "desc", ["repo"], [], "low", "low")

    def test_get_report(self):
        analyzer = ImpactAnalyzer()
        report = analyzer.compile_impact_report("a", "desc", ["r"], ["/e"], "low", "low")
        fetched = analyzer.get_report(report.report_id)
        assert fetched.report_id == report.report_id

    def test_get_nonexistent_raises(self):
        analyzer = ImpactAnalyzer()
        with pytest.raises(KeyError):
            analyzer.get_report("nonexistent")
