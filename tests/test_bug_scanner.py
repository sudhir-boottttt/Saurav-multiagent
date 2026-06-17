"""Property tests and unit tests for the Bug Scanner.

Property tests cover Properties 20-21 from the design document.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.bug_scanner import BugScanner, ScanResult
from src.core.enums import BugSeverity
from src.core.models import BugReport
from tests.generators import bug_report_st, uuid_st, non_empty_str


# ──────────────────────────────────────────────
# Property 20: Bug Report Completeness and Severity Routing
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 20: Bug Report Completeness and Severity Routing


@given(data=bug_report_st)
@settings(max_examples=100)
def test_property_20_bug_report_completeness(data):
    """Bug reports SHALL contain file_path, line_start, line_end, category,
    severity, and description (all non-empty). CRITICAL/HIGH triggers fix
    generation; MEDIUM/LOW does not.
    """
    # Ensure line_end >= line_start
    if data["line_end"] < data["line_start"]:
        data["line_start"], data["line_end"] = data["line_end"], data["line_start"]

    bug = BugReport(**data)

    # Validate completeness
    assert bug.file_path, "file_path must be non-empty"
    assert bug.line_start >= 1, "line_start must be >= 1"
    assert bug.line_end >= bug.line_start, "line_end must be >= line_start"
    assert bug.category, "category must be non-empty"
    assert bug.severity is not None, "severity must be set"
    assert bug.description, "description must be non-empty"

    # Validate severity routing
    route = BugScanner.route_by_severity(bug)
    if bug.severity in (BugSeverity.CRITICAL, BugSeverity.HIGH):
        assert route == "fix_and_discuss"
    else:
        assert route == "log_and_track"


# ──────────────────────────────────────────────
# Property 21: Approved Fix to PR Pipeline
# ──────────────────────────────────────────────
# Feature: multi-agent-platform, Property 21: Approved Fix to PR Pipeline


@given(
    file_path=st.builds(lambda n: f"src/{n}.py", n=non_empty_str),
    content=non_empty_str,
    repo=non_empty_str,
)
@settings(max_examples=100)
def test_property_21_scan_report_validation(file_path, content, repo):
    """All generated bug reports SHALL pass validation (non-empty required fields)."""
    scanner = BugScanner(agent_id="test-scanner")
    bugs = scanner.scan_content(content, file_path, repo)

    for bug in bugs:
        assert BugScanner.validate_report(bug), (
            f"Bug report failed validation: {bug.report_id}"
        )
        assert bug.agent_id == "test-scanner"
        assert bug.repo == repo
        assert bug.file_path == file_path


# ──────────────────────────────────────────────
# Unit Tests
# ──────────────────────────────────────────────


class TestBugScanner:

    def test_detect_bare_except_pass(self):
        code = '''
try:
    do_something()
except:
    pass
'''
        scanner = BugScanner()
        bugs = scanner.scan_content(code, "src/main.py", "my-repo")
        assert len(bugs) >= 1
        assert any(b.category == "unhandled_exception" for b in bugs)

    def test_detect_exception_pass(self):
        code = '''
try:
    do_something()
except Exception:
    pass
'''
        scanner = BugScanner()
        bugs = scanner.scan_content(code, "src/main.py", "my-repo")
        assert len(bugs) >= 1

    def test_clean_code_no_bugs(self):
        code = '''
def add(a: int, b: int) -> int:
    return a + b
'''
        scanner = BugScanner()
        bugs = scanner.scan_content(code, "src/clean.py", "my-repo")
        assert len(bugs) == 0

    def test_scan_files_multiple(self):
        files = {
            "src/a.py": "try:\n    x()\nexcept:\n    pass\n",
            "src/b.py": "def clean(): return 1\n",
        }
        scanner = BugScanner()
        result = scanner.scan_files(files, "my-repo")
        assert result.files_scanned == 2
        assert not result.clean
        assert len(result.bugs) >= 1

    def test_scan_files_all_clean(self):
        files = {
            "src/a.py": "def f(): return 1\n",
            "src/b.py": "x = 42\n",
        }
        scanner = BugScanner()
        result = scanner.scan_files(files, "my-repo")
        assert result.clean
        assert len(result.bugs) == 0

    def test_severity_routing_critical(self):
        bug = BugReport(
            agent_id="dev-001", repo="svc", file_path="f.py",
            line_start=1, line_end=1, category="null_reference",
            severity=BugSeverity.CRITICAL, description="null ref",
        )
        assert BugScanner.route_by_severity(bug) == "fix_and_discuss"

    def test_severity_routing_low(self):
        bug = BugReport(
            agent_id="dev-001", repo="svc", file_path="f.py",
            line_start=1, line_end=1, category="style",
            severity=BugSeverity.LOW, description="style issue",
        )
        assert BugScanner.route_by_severity(bug) == "log_and_track"

    def test_validate_report_valid(self):
        bug = BugReport(
            agent_id="dev-001", repo="svc", file_path="f.py",
            line_start=1, line_end=5, category="null_reference",
            severity=BugSeverity.HIGH, description="found null ref",
        )
        assert BugScanner.validate_report(bug)

    def test_bug_report_has_agent_id(self):
        scanner = BugScanner(agent_id="my-agent")
        code = "try:\n    x()\nexcept:\n    pass\n"
        bugs = scanner.scan_content(code, "f.py", "repo")
        for bug in bugs:
            assert bug.agent_id == "my-agent"
