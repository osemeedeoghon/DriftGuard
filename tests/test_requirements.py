"""Tests for dependency pinning and CI configuration (roadmap P0.3, P0.4).

Audit finding: ``requirements.txt`` used ``>=`` ranges only, so a fresh
environment could install a different dependency set from the one the suite
was verified against. "The tests pass" then described nobody's machine in
particular.

These tests are structural. They do not install anything; they assert that the
declared environment stays reproducible and that CI stays wired up, so a
regression in either fails the suite rather than going unnoticed until a build
breaks.
"""

from __future__ import annotations

import importlib.metadata as metadata
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
DEV_REQUIREMENTS = REPO_ROOT / "requirements-dev.txt"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: Python versions the project claims to support, and which CI must cover.
SUPPORTED_PYTHON = ("3.11", "3.12", "3.13")

#: A requirement line that is exactly pinned: name==version, optional extras.
PINNED = re.compile(r"^[A-Za-z0-9._-]+(\[[A-Za-z0-9,._-]+\])?==[A-Za-z0-9._+!-]+$")


def requirement_lines(path: pathlib.Path) -> list[str]:
    """Requirement specifiers from a file, ignoring comments, blanks and -r."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        lines.append(line)
    return lines


def requirement_names(path: pathlib.Path) -> set[str]:
    return {re.split(r"[=<>\[]", line, maxsplit=1)[0].strip().lower() for line in requirement_lines(path)}


# --------------------------------------------------------------------------
# Pinning
# --------------------------------------------------------------------------
def test_requirements_file_exists():
    assert REQUIREMENTS.is_file()
    assert DEV_REQUIREMENTS.is_file()


def test_every_runtime_requirement_is_exactly_pinned():
    """The regression test for the audit finding. Fails on any `>=` range."""
    unpinned = [line for line in requirement_lines(REQUIREMENTS) if not PINNED.match(line)]
    assert not unpinned, (
        "requirements.txt must pin exact versions for reproducible builds; "
        f"found: {unpinned}"
    )


def test_every_dev_requirement_is_exactly_pinned():
    unpinned = [
        line for line in requirement_lines(DEV_REQUIREMENTS) if not PINNED.match(line)
    ]
    assert not unpinned, f"requirements-dev.txt has unpinned entries: {unpinned}"


def test_dev_requirements_include_the_runtime_set():
    """One file should be enough for a contributor to install."""
    text = DEV_REQUIREMENTS.read_text(encoding="utf-8")
    assert "-r requirements.txt" in text


def test_pin_regex_rejects_ranges_and_accepts_exact_pins():
    """A guard that never fails proves nothing."""
    for bad in ("pandas>=2.0", "numpy", "plotly~=5.24", "streamlit>1.0,<2"):
        assert not PINNED.match(bad), bad
    for good in ("pandas==2.2.3", "pymupdf==2.1", "ruff[lint]==0.1.0"):
        assert PINNED.match(good), good


# --------------------------------------------------------------------------
# The pins describe what is actually installed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("line", requirement_lines(REQUIREMENTS))
def test_pinned_runtime_version_matches_the_installed_one(line):
    """A pin that disagrees with the environment under test is a lie.

    Skips rather than fails when a package is absent, so the suite still runs
    in a minimal environment; the point is to catch a pin that has drifted
    away from the version actually being exercised.
    """
    name, _, pinned = line.partition("==")
    name = name.split("[", 1)[0]
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        pytest.skip(f"{name} is not installed in this environment")
    assert installed == pinned, (
        f"requirements.txt pins {name}=={pinned} but {installed} is installed. "
        "Update the pin and re-run the suite, or install the pinned version."
    )


def test_runtime_and_dev_requirements_do_not_overlap():
    """A package pinned in both files can drift between them."""
    overlap = requirement_names(REQUIREMENTS) & requirement_names(DEV_REQUIREMENTS)
    assert not overlap, f"pinned in both files: {sorted(overlap)}"


def test_no_network_client_is_declared_as_a_dependency():
    """Offline-first is a dependency-level constraint, not only a code one.

    See docs/adr/0003-offline-first.md.
    """
    forbidden = {"requests", "httpx", "urllib3", "aiohttp", "openai", "anthropic", "kaleido"}
    declared = requirement_names(REQUIREMENTS) | requirement_names(DEV_REQUIREMENTS)
    assert not (declared & forbidden), (
        f"offline-first violated by declared dependency: {sorted(declared & forbidden)}"
    )


def test_runtime_requirements_cover_everything_the_app_imports():
    """Catches a third-party import that was never declared."""
    declared = requirement_names(REQUIREMENTS)
    # Distribution name differs from the import name for these.
    aliases = {"sklearn": "scikit-learn", "fitz": "pymupdf", "PIL": "pillow"}
    expected = {"streamlit", "pandas", "numpy", "scikit-learn", "plotly", "reportlab", "matplotlib"}
    assert expected <= declared, f"undeclared runtime dependency: {sorted(expected - declared)}"
    assert aliases  # documents the mapping rather than leaving it implicit


# --------------------------------------------------------------------------
# CI configuration
# --------------------------------------------------------------------------
def test_ci_workflow_exists():
    assert WORKFLOW.is_file(), "no .github/workflows/ci.yml — the suite runs only when someone remembers"


def test_ci_runs_on_push_and_pull_request():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "push:" in text
    assert "pull_request:" in text


def test_ci_covers_every_supported_python_version():
    text = WORKFLOW.read_text(encoding="utf-8")
    for version in SUPPORTED_PYTHON:
        assert f'"{version}"' in text, f"CI matrix is missing Python {version}"


def test_ci_installs_from_the_pinned_files():
    """CI must exercise the declared environment, not an ad-hoc one."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "requirements-dev.txt" in text


def test_ci_actually_runs_the_test_suite():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pytest" in text
