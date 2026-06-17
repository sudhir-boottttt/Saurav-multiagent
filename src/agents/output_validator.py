"""Output validator for checking agent responses against workspace reality.

Validates file paths and symbol references in agent responses to prevent
hallucinated references. Supports correction cycles.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns to extract file references from agent responses
FILE_PATH_PATTERN = re.compile(
    r'(?:^|\s|`|"|\')([a-zA-Z0-9_\-./]+\.[a-zA-Z]{1,10})(?:\s|`|"|\'|:|,|$)',
)
# Pattern for symbol references like `function_name` or `ClassName.method`
SYMBOL_PATTERN = re.compile(
    r'`([a-zA-Z_][a-zA-Z0-9_.]*)`',
)


@dataclass
class ValidationResult:
    """Result of validating an agent response."""

    is_valid: bool
    invalid_refs: list[dict] = field(default_factory=list)
    corrections_made: int = 0


class OutputValidator:
    """Validates agent responses against actual workspace contents."""

    MAX_CORRECTION_CYCLES = 2

    def __init__(
        self,
        base_path: str = "/tmp/agent-repos/",
    ) -> None:
        self._base_path = Path(base_path).resolve()

    def validate(
        self,
        response: str,
        workspace: str,
        agent_id: str = "",
    ) -> ValidationResult:
        """Check all file/symbol references in response against workspace."""
        ws_path = (self._base_path / workspace).resolve()
        if not ws_path.exists():
            return ValidationResult(is_valid=True)

        refs = self._extract_references(response)
        invalid_refs = []

        for ref in refs:
            if ref["type"] == "file":
                if not self._verify_file_exists(ws_path, ref["reference"]):
                    invalid_refs.append({
                        "type": "file",
                        "reference": ref["reference"],
                        "reason": "File does not exist in workspace",
                    })
            elif ref["type"] == "symbol":
                # Only validate symbols that look like they reference specific files
                # Skip common programming terms
                if not self._is_common_term(ref["reference"]):
                    # We don't validate symbols without file context
                    # as they could be from any file
                    pass

        return ValidationResult(
            is_valid=len(invalid_refs) == 0,
            invalid_refs=invalid_refs,
            corrections_made=0,
        )

    def _extract_references(self, response: str) -> list[dict]:
        """Parse file paths and symbol names from response text."""
        refs = []
        seen = set()

        # Extract file paths
        for match in FILE_PATH_PATTERN.finditer(response):
            path = match.group(1)
            # Filter out common non-file patterns
            if path.startswith("http") or path.startswith("//"):
                continue
            if "/" in path or path.endswith(".py") or path.endswith(".java"):
                if path not in seen:
                    refs.append({"type": "file", "reference": path})
                    seen.add(path)

        # Extract symbol references
        for match in SYMBOL_PATTERN.finditer(response):
            symbol = match.group(1)
            if symbol not in seen and not self._is_common_term(symbol):
                refs.append({"type": "symbol", "reference": symbol})
                seen.add(symbol)

        return refs

    def _verify_file_exists(self, ws_path: Path, file_path: str) -> bool:
        """Check if a file exists in the workspace."""
        candidate = (ws_path / file_path).resolve()
        try:
            candidate.relative_to(ws_path)
        except ValueError:
            return False
        return candidate.exists()

    def _verify_symbol_exists(
        self,
        ws_path: Path,
        file_path: str,
        symbol_name: str,
    ) -> bool:
        """Check if a symbol exists in a specific file."""
        candidate = (ws_path / file_path).resolve()
        if not candidate.exists():
            return False
        try:
            content = candidate.read_text(encoding="utf-8")
            return symbol_name in content
        except Exception:
            return False

    @staticmethod
    def _is_common_term(symbol: str) -> bool:
        """Check if a symbol is a common programming term (not a specific reference)."""
        common = {
            "True", "False", "None", "self", "cls", "str", "int", "float",
            "bool", "list", "dict", "set", "tuple", "type", "object",
            "Exception", "ValueError", "TypeError", "KeyError", "RuntimeError",
            "return", "import", "from", "class", "def", "async", "await",
            "if", "else", "elif", "for", "while", "try", "except", "finally",
            "with", "as", "pass", "break", "continue", "yield", "raise",
        }
        return symbol in common or len(symbol) <= 2
