"""Bug detection and fix generation pipeline.

Scans code for bug patterns (null refs, unhandled exceptions, API
mismatches, dependency conflicts), generates reports with severity
routing, and manages the fix review loop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4

from src.core.enums import BugSeverity
from src.core.models import BugReport

logger = logging.getLogger(__name__)


# Bug pattern definitions
BUG_PATTERNS = {
    "null_reference": [
        re.compile(r"\bNone\b.*\.\w+"),  # Accessing attribute on None
        re.compile(r"if\s+\w+\s+is\s+not\s+None.*\n.*else.*\.\w+"),  # Missing None guard
        re.compile(r"\.get\([^)]+\)\.\w+"),  # Chained access after .get() without default
    ],
    "unhandled_exception": [
        re.compile(r"except\s*:\s*\n\s*pass"),  # Bare except with pass
        re.compile(r"except\s+Exception\s*:\s*\n\s*pass"),  # Catch-all with pass
    ],
    "api_mismatch": [
        re.compile(r"requests\.(get|post|put|delete)\([^)]*\)\s*\n(?!.*\.status_code)"),
        re.compile(r"\.json\(\)(?!\s*#\s*validated)"),  # Unvalidated JSON parse
    ],
    "dependency_conflict": [
        re.compile(r"import\s+(\w+).*#\s*deprecated"),
        re.compile(r"from\s+(\w+)\s+import.*#\s*breaking"),
    ],
}

SEVERITY_THRESHOLDS = {
    "null_reference": BugSeverity.HIGH,
    "unhandled_exception": BugSeverity.MEDIUM,
    "api_mismatch": BugSeverity.HIGH,
    "dependency_conflict": BugSeverity.CRITICAL,
}


@dataclass
class ScanResult:
    """Result of a bug scan on a file or repo."""
    bugs: list[BugReport] = field(default_factory=list)
    files_scanned: int = 0
    clean: bool = True


class BugScanner:
    """Scans code for common bug patterns and generates reports."""

    def __init__(self, agent_id: str = "scanner") -> None:
        self._agent_id = agent_id

    def scan_content(
        self, content: str, file_path: str, repo: str,
        severity_override: Optional[BugSeverity] = None,
    ) -> list[BugReport]:
        """Scan a single file's content for bug patterns.

        Returns list of BugReport for each detected issue.
        """
        bugs = []
        lines = content.split("\n")

        for category, patterns in BUG_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(content):
                    start_pos = match.start()
                    line_start = content[:start_pos].count("\n") + 1
                    line_end = line_start + match.group().count("\n")

                    severity = severity_override or SEVERITY_THRESHOLDS.get(
                        category, BugSeverity.LOW,
                    )

                    bug = BugReport(
                        report_id=str(uuid4()),
                        agent_id=self._agent_id,
                        repo=repo,
                        file_path=file_path,
                        line_start=line_start,
                        line_end=max(line_end, line_start),
                        category=category,
                        severity=severity,
                        description=f"Detected {category} pattern: {match.group()[:80]}",
                    )
                    bugs.append(bug)

        return bugs

    def scan_files(
        self, files: dict[str, str], repo: str,
    ) -> ScanResult:
        """Scan multiple files. files = {file_path: content}.

        Returns ScanResult with all detected bugs.
        """
        result = ScanResult(files_scanned=len(files))
        for file_path, content in files.items():
            bugs = self.scan_content(content, file_path, repo)
            result.bugs.extend(bugs)
        result.clean = len(result.bugs) == 0
        return result

    @staticmethod
    def route_by_severity(bug: BugReport) -> str:
        """Determine routing action based on severity.

        Returns "fix_and_discuss" for CRITICAL/HIGH, "log_and_track" for MEDIUM/LOW.
        """
        if bug.severity in (BugSeverity.CRITICAL, BugSeverity.HIGH):
            return "fix_and_discuss"
        return "log_and_track"

    @staticmethod
    def validate_report(bug: BugReport) -> bool:
        """Validate that a bug report has all required fields non-empty."""
        return all([
            bug.file_path,
            bug.line_start >= 1,
            bug.line_end >= bug.line_start,
            bug.category,
            bug.severity,
            bug.description,
        ])
