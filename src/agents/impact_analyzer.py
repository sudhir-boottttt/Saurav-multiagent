"""Cross-domain impact analysis workflow.

Detects shared API/data model changes, queries affected domains,
compiles impact reports, and triggers discussion threads for
breaking changes.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4

from src.core.models import ImpactReport

logger = logging.getLogger(__name__)


class ImpactAnalyzer:
    """Analyzes cross-domain impact of proposed changes."""

    def __init__(self) -> None:
        self._reports: dict[str, ImpactReport] = {}

    def detect_cross_domain_change(
        self,
        changed_files: list[str],
        shared_patterns: list[str] | None = None,
    ) -> bool:
        """Detect if changed files affect shared APIs or data models.

        shared_patterns: list of path patterns considered shared
        (e.g., ["shared/", "contracts/", "proto/"]).
        """
        patterns = shared_patterns or ["shared/", "contracts/", "proto/", "api/"]
        for f in changed_files:
            if any(p in f for p in patterns):
                return True
        return False

    def compile_impact_report(
        self,
        trigger_agent_id: str,
        change_description: str,
        affected_repos: list[str],
        affected_endpoints: list[str],
        estimated_effort: str,
        risk_level: str,
        has_breaking_changes: bool = False,
    ) -> ImpactReport:
        """Compile a cross-domain impact report.

        All fields must be non-empty per Property 23.
        """
        if not affected_repos:
            raise ValueError("affected_repos must be non-empty")
        if not affected_endpoints:
            raise ValueError("affected_endpoints must be non-empty")
        if not estimated_effort:
            raise ValueError("estimated_effort must be non-empty")
        if not risk_level:
            raise ValueError("risk_level must be non-empty")

        report = ImpactReport(
            report_id=str(uuid4()),
            trigger_agent_id=trigger_agent_id,
            change_description=change_description,
            affected_repos=affected_repos,
            affected_endpoints=affected_endpoints,
            estimated_effort=estimated_effort,
            risk_level=risk_level,
            has_breaking_changes=has_breaking_changes,
        )
        self._reports[report.report_id] = report
        logger.info("Compiled impact report %s (breaking=%s)", report.report_id, has_breaking_changes)
        return report

    def get_report(self, report_id: str) -> ImpactReport:
        if report_id not in self._reports:
            raise KeyError(f"Impact report {report_id} not found")
        return self._reports[report_id]

    @staticmethod
    def validate_report(report: ImpactReport) -> bool:
        """Validate report completeness per Property 23."""
        return all([
            report.affected_repos,
            report.affected_endpoints,
            report.estimated_effort,
            report.risk_level,
            report.trigger_agent_id,
            report.change_description,
        ])
