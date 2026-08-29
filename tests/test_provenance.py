"""Tests for source and revision provenance."""

from __future__ import annotations

import hashlib

from fairness_project.provenance import git_state, source_sha256


def test_source_fingerprint_is_stable_and_nonempty() -> None:
    fingerprint = source_sha256()

    assert fingerprint == source_sha256()
    assert len(fingerprint) == 64
    assert fingerprint != hashlib.sha256(b"").hexdigest()


def test_git_state_is_anchored_to_the_owning_checkout() -> None:
    state = git_state()

    assert len(state.commit) == 40
    assert isinstance(state.dirty_worktree, bool)
