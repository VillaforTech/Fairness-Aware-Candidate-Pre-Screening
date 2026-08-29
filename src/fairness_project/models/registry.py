"""Read-only discovery for versioned experiment bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fairness_project.models.artifact import MANIFEST_FILENAME, ModelBundle, load_bundle


class ModelRegistry:
    """Discover and load bundles created by :func:`run_experiment`."""

    def __init__(self, runs_dir: str | Path = "runs"):
        self.runs_dir = Path(runs_dir)

    def list_runs(self) -> list[str]:
        if not self.runs_dir.is_dir():
            return []
        candidates = [
            path
            for path in self.runs_dir.iterdir()
            if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
        ]
        candidates.sort(
            key=lambda path: (path / MANIFEST_FILENAME).stat().st_mtime,
            reverse=True,
        )
        return [path.name for path in candidates]

    def latest_run_id(self) -> str | None:
        runs = self.list_runs()
        return runs[0] if runs else None

    def load_bundle(self, run_id: str | None = None) -> ModelBundle:
        resolved = run_id
        if resolved in (None, "latest"):
            resolved = self.latest_run_id()
        if resolved is None:
            raise FileNotFoundError(f"No validated run bundles found in {self.runs_dir}")
        return load_bundle(self.runs_dir / resolved)

    def load_model(self, run_id: str | None = None) -> Any:
        return self.load_bundle(run_id).model

    def load_metadata(self, run_id: str | None = None) -> dict[str, Any]:
        return self.load_bundle(run_id).manifest


def load_model(run_id: str | None = None, runs_dir: str | Path = "runs") -> Any:
    """Compatibility helper backed by the validated bundle loader."""
    return ModelRegistry(runs_dir).load_model(run_id)
