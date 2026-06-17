"""Tests for CodeAnalyzer."""

import pytest

from src.code_ops.code_analyzer import CodeAnalyzer


@pytest.fixture
def analyzer(tmp_path):
    a = CodeAnalyzer(base_path=str(tmp_path))
    # Create workspace with test files
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "main.py").write_text(
        'from utils import helper\n\n'
        'def main():\n'
        '    result = helper("test")\n'
        '    return result\n'
    )
    (ws / "utils.py").write_text(
        'def helper(name: str) -> str:\n'
        '    """Help function."""\n'
        '    return f"Hello, {name}"\n'
    )
    (ws / "data.txt").write_text("some data\nfoo bar\nbaz\n")
    return a


def test_read_file(analyzer):
    content = analyzer.read_file("repo", "main.py")
    assert "def main():" in content


def test_read_file_not_found(analyzer):
    with pytest.raises(FileNotFoundError):
        analyzer.read_file("repo", "nonexistent.py")


def test_path_traversal_blocked(analyzer):
    with pytest.raises(PermissionError):
        analyzer.read_file("repo", "../../../etc/passwd")


def test_grep_finds_matches(analyzer):
    matches = analyzer.grep("repo", "helper", "*.py")
    assert len(matches) >= 2  # definition + usage
    assert all(m.line_content for m in matches)
    assert all(m.line_number >= 1 for m in matches)


def test_grep_with_context(analyzer):
    matches = analyzer.grep("repo", "def main", "*.py")
    assert len(matches) >= 1
    match = matches[0]
    assert len(match.context_after) > 0


def test_grep_file_glob_filter(analyzer):
    matches = analyzer.grep("repo", "data", "*.txt")
    assert len(matches) >= 1
    # Should not match .py files
    for m in matches:
        assert m.file_path.endswith(".txt")


def test_grep_no_matches(analyzer):
    matches = analyzer.grep("repo", "zzz_nonexistent_zzz", "*.py")
    assert matches == []


def test_find_references(analyzer):
    refs = analyzer.find_references("repo", "helper")
    types = {r.ref_type for r in refs}
    assert "definition" in types
    assert "call" in types or "import" in types


def test_find_references_import(analyzer):
    refs = analyzer.find_references("repo", "helper")
    import_refs = [r for r in refs if r.ref_type == "import"]
    assert len(import_refs) >= 1


def test_get_ast(analyzer):
    tree = analyzer.get_ast("repo", "main.py")
    assert tree.node_type == "Module"
    # Should have function children
    func_names = [c.name for c in tree.children]
    assert "main" in func_names


# Property 9: File Read/Write Round-Trip
def test_read_write_round_trip(analyzer, tmp_path):
    """Read via CodeAnalyzer then write back produces identical file."""
    from src.code_ops.local_file_manager import LocalFileManager

    lfm = LocalFileManager(base_path=str(tmp_path))
    original = analyzer.read_file("repo", "main.py")
    lfm.write_file("repo", "main.py", original)
    after = analyzer.read_file("repo", "main.py")
    assert after == original
