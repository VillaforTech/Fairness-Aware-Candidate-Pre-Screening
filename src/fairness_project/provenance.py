"""Stable source and revision provenance for experiments."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

UNAVAILABLE_GIT_COMMIT = "unavailable"


@dataclass(frozen=True)
class GitState:
    """Git revision information when the installed code is a checkout."""

    commit: str
    dirty_worktree: bool | None


def _source_checkout_root() -> Path | None:
    """Return the repository root only when this module is loaded from its source tree."""
    module_path = Path(__file__).resolve()
    for candidate in module_path.parents:
        expected = candidate / "src" / "fairness_project" / module_path.name
        if expected.resolve() == module_path and (candidate / ".git").exists():
            return candidate
    return None


def git_state() -> GitState:
    """Read Git state from the owning checkout, never from the caller's directory."""
    root = _source_checkout_root()
    if root is None:
        return GitState(UNAVAILABLE_GIT_COMMIT, None)
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return GitState(UNAVAILABLE_GIT_COMMIT, None)
    return GitState(commit, dirty)


def source_sha256() -> str:
    """Hash the canonical package source identically in a checkout or installed wheel."""
    package_root = Path(__file__).resolve().parent
    candidates = sorted(package_root.rglob("*.py"))
    typed_marker = package_root / "py.typed"
    if typed_marker.is_file():
        candidates.append(typed_marker)

    digest = hashlib.sha256()
    for path in sorted(candidates):
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
