"""Check that required documentation files exist and key modules have docstrings."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REQUIRED_DOCS = [
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "docs/architecture.md",
    "docs/methodology.md",
    "docs/references.md",
    "docs/data.md",
    "docs/model_card.md",
    "docs/responsible_ai.md",
    "docs/api_spec.md",
    "docs/governance.md",
    "docs/deployment.md",
    "docs/monitoring.md",
]

MODULES_REQUIRING_DOCSTRINGS = [
    "src/fairness_project/config.py",
    "src/fairness_project/cli.py",
    "src/fairness_project/inference/api.py",
    "src/fairness_project/governance/gate.py",
    "src/fairness_project/data/quality.py",
    "src/fairness_project/evaluation/intersectional.py",
    "src/fairness_project/evaluation/overlap.py",
    "src/fairness_project/evaluation/stability.py",
    "src/fairness_project/fairness/frontier.py",
    "src/fairness_project/fairness/selective.py",
    "src/fairness_project/monitoring/snapshot.py",
    "src/fairness_project/reporting/html.py",
]


def check_doc_files() -> list[str]:
    """Return list of missing or empty doc files."""
    missing = []
    for doc in REQUIRED_DOCS:
        path = Path(doc)
        if not path.exists():
            missing.append(f"MISSING: {doc}")
        elif path.stat().st_size == 0:
            missing.append(f"EMPTY: {doc}")
    return missing


def check_module_docstrings() -> list[str]:
    """Return list of modules missing module-level docstrings."""
    missing = []
    for module_path in MODULES_REQUIRING_DOCSTRINGS:
        path = Path(module_path)
        if not path.exists():
            missing.append(f"MODULE NOT FOUND: {module_path}")
            continue
        try:
            tree = ast.parse(path.read_text())
            docstring = ast.get_docstring(tree)
            if not docstring:
                missing.append(f"NO DOCSTRING: {module_path}")
        except SyntaxError:
            missing.append(f"SYNTAX ERROR: {module_path}")
    return missing


def main() -> int:
    """Run all documentation coverage checks."""
    errors = []

    print("Checking required documentation files...")
    doc_errors = check_doc_files()
    errors.extend(doc_errors)
    for err in doc_errors:
        print(f"  {err}")

    print("Checking module docstrings...")
    docstring_errors = check_module_docstrings()
    errors.extend(docstring_errors)
    for err in docstring_errors:
        print(f"  {err}")

    if errors:
        print(f"\nDoc coverage check FAILED: {len(errors)} issue(s) found")
        return 1

    print("\nDoc coverage check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
