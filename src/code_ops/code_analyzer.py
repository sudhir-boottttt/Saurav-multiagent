"""Code analyzer for direct file access, grep, symbol references, and AST parsing.

Provides agents with precise code navigation tools beyond vector search.
"""

from __future__ import annotations

import ast
import fnmatch
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GrepMatch:
    """A single grep match with context lines."""

    file_path: str
    line_number: int
    line_content: str
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)


@dataclass
class Reference:
    """A symbol reference found via AST analysis."""

    file_path: str
    line_number: int
    ref_type: str  # "definition" | "call" | "import"
    symbol_name: str


@dataclass
class ASTNode:
    """A simplified AST node representation."""

    node_type: str
    name: Optional[str]
    start_line: int
    end_line: int
    children: list[ASTNode] = field(default_factory=list)


class CodeAnalyzer:
    """Direct file access, grep, symbol references, and AST parsing."""

    CONTEXT_LINES = 3

    def __init__(self, base_path: str = "/tmp/agent-repos/") -> None:
        self._base_path = Path(base_path).resolve()

    def _resolve_safe_path(self, workspace: str, file_path: str) -> Path:
        """Resolve path within workspace, reject traversal."""
        ws_path = (self._base_path / workspace).resolve()
        if Path(file_path).is_absolute():
            candidate = Path(file_path).resolve()
        else:
            candidate = (ws_path / file_path).resolve()

        try:
            candidate.relative_to(ws_path)
        except ValueError:
            raise PermissionError(
                f"Path traversal blocked: '{file_path}' escapes workspace '{workspace}'"
            )
        return candidate

    def _workspace_path(self, workspace: str) -> Path:
        return (self._base_path / workspace).resolve()

    def read_file(self, workspace: str, file_path: str) -> str:
        """Read full file contents within workspace."""
        safe_path = self._resolve_safe_path(workspace, file_path)
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return safe_path.read_text(encoding="utf-8")

    def grep(
        self,
        workspace: str,
        pattern: str,
        file_glob: str = "*",
    ) -> list[GrepMatch]:
        """Regex search with file glob filtering and context lines."""
        ws_path = self._workspace_path(workspace)
        if not ws_path.exists():
            return []

        regex = re.compile(pattern)
        matches = []

        for fp in ws_path.rglob("*"):
            if not fp.is_file():
                continue
            rel = str(fp.relative_to(ws_path))
            if not fnmatch.fnmatch(fp.name, file_glob):
                continue

            try:
                lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue

            for i, line in enumerate(lines):
                if regex.search(line):
                    ctx_before = lines[max(0, i - self.CONTEXT_LINES):i]
                    ctx_after = lines[i + 1:i + 1 + self.CONTEXT_LINES]
                    matches.append(GrepMatch(
                        file_path=rel,
                        line_number=i + 1,
                        line_content=line,
                        context_before=ctx_before,
                        context_after=ctx_after,
                    ))

        return matches

    def find_references(
        self,
        workspace: str,
        symbol_name: str,
    ) -> list[Reference]:
        """Find all usages of a symbol via AST parsing (Python files)."""
        ws_path = self._workspace_path(workspace)
        if not ws_path.exists():
            return []

        refs = []
        for fp in ws_path.rglob("*.py"):
            try:
                source = fp.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source)
            except (SyntaxError, Exception):
                continue

            rel = str(fp.relative_to(ws_path))
            for node in ast.walk(tree):
                # Definitions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == symbol_name:
                        refs.append(Reference(rel, node.lineno, "definition", symbol_name))
                elif isinstance(node, ast.ClassDef):
                    if node.name == symbol_name:
                        refs.append(Reference(rel, node.lineno, "definition", symbol_name))
                # Calls
                elif isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id == symbol_name:
                        refs.append(Reference(rel, node.lineno, "call", symbol_name))
                    elif isinstance(func, ast.Attribute) and func.attr == symbol_name:
                        refs.append(Reference(rel, node.lineno, "call", symbol_name))
                # Imports
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name == symbol_name:
                            refs.append(Reference(rel, node.lineno, "import", symbol_name))

        return refs

    def get_ast(self, workspace: str, file_path: str) -> ASTNode:
        """Parse file into structured AST representation."""
        safe_path = self._resolve_safe_path(workspace, file_path)
        source = safe_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        return self._convert_node(tree)

    def _convert_node(self, node: ast.AST) -> ASTNode:
        """Convert an ast.AST node to our simplified ASTNode."""
        name = getattr(node, "name", None)
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        node_type = type(node).__name__

        children = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                children.append(self._convert_node(child))

        return ASTNode(
            node_type=node_type,
            name=name,
            start_line=start,
            end_line=end,
            children=children,
        )
