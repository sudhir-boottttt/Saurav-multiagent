"""Tests for OutputValidator."""

import pytest

from src.agents.output_validator import OutputValidator


@pytest.fixture
def validator(tmp_path):
    v = OutputValidator(base_path=str(tmp_path))
    # Create workspace with files
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "main.py").write_text("def main(): pass\n")
    (ws / "src" / "utils.py").write_text("def helper(): pass\n")
    (ws / "README.md").write_text("# Project\n")
    return v


def test_valid_response(validator):
    """Response referencing existing files should pass."""
    response = "The fix is in src/main.py where we modify the main function."
    result = validator.validate(response, "repo")
    assert result.is_valid is True


def test_invalid_file_reference(validator):
    """Response referencing non-existent file should fail."""
    response = "Check the file src/nonexistent.py for the bug."
    result = validator.validate(response, "repo")
    assert result.is_valid is False
    assert len(result.invalid_refs) >= 1
    assert result.invalid_refs[0]["type"] == "file"


def test_no_file_references(validator):
    """Response without file references should pass."""
    response = "The issue is related to memory management."
    result = validator.validate(response, "repo")
    assert result.is_valid is True


def test_nonexistent_workspace(validator):
    """Non-existent workspace should pass (nothing to validate against)."""
    result = validator.validate("Check src/main.py", "nonexistent-ws")
    assert result.is_valid is True


def test_multiple_valid_refs(validator):
    response = "Modified src/main.py and src/utils.py to fix the issue."
    result = validator.validate(response, "repo")
    assert result.is_valid is True


def test_mixed_valid_invalid_refs(validator):
    response = "Fixed src/main.py but also need to check src/missing.py"
    result = validator.validate(response, "repo")
    assert result.is_valid is False


# Property 18: Output Validation Completeness
def test_passing_validation_means_all_refs_exist(validator):
    """If validation passes, every file ref must exist."""
    response = "Updated src/main.py and README.md"
    result = validator.validate(response, "repo")
    assert result.is_valid is True
    assert len(result.invalid_refs) == 0


def test_extract_references_from_code_blocks(validator):
    """Should extract file paths from code block references."""
    response = "Edit `src/main.py` to add the new function."
    result = validator.validate(response, "repo")
    assert result.is_valid is True


def test_ignores_urls(validator):
    """Should not treat URLs as file references."""
    response = "See https://example.com/path/file.py for details."
    result = validator.validate(response, "repo")
    assert result.is_valid is True
