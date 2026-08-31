# Changelog

All notable changes are recorded here. Release numbers follow semantic
versioning when a release is tagged.

## Unreleased

### Added

- Validation-only two-dimensional group-threshold search with a deterministic
  accuracy versus absolute TPR-gap Pareto frontier.
- Global probability review band with held-out coverage, error, and group-burden
  diagnostics.
- Sex by original-race diagnostics with Wilson intervals, calibration,
  effective sample size, evidence states, and worst-group spans.
- Weighted held-out sensitivity using `fnlwgt` outside the model feature set.
- Exact-feature overlap sensitivity with fixed-policy metrics on the complete
  and overlap-excluded held-out slices.
- Repeated-seed stability studies with strict comparability checks and
  worst-seed reporting.
- A digest-bound preprocessing quality sidecar with raw and processed data
  semantics, including 7.41% complete-case attrition, duplicate rows, repeated
  11-feature vectors, conflicting labels, and cross-split overlap.
- Self-contained HTML audit reports bound into each run artifact.
- A strict aggregate-only monitoring reference in every run bundle, plus
  `fairness monitor snapshot` and `fairness monitor compare` for offline drift
  evidence with `PASS`, `FAIL`, and `INSUFFICIENT_EVIDENCE` outcomes.
- v2 local simulation API and explicit research override for rejected bundles.
- Architecture, methodology, governance, data, security, and contribution
  documentation.

### Changed

- Reframed the project as the Auditable Fair-ML Policy Lab.
- Reduced the scoring contract from 12 to 11 features by removing the Census
  final weight.
- Upgraded configuration, report, artifact, and policy contracts to schema 2.0.
- Expanded the evidence gate to include FPR gap, bootstrap bounds,
  intersectional spans, and held-out review behavior.
- Made seven-file bundle publication atomic and bound model, report, policy,
  predictions, HTML, and monitoring evidence by SHA-256.
- Bound each experiment to its verified data-semantics evidence and recorded the
  quality-sidecar digest when the canonical sidecar is available.
- Replaced prediction wording with evaluation-only simulation wording.

### Removed

- Stale notebook, flat-script, and alternate training paths that contradicted
  the canonical split and feature contracts.

## 0.2.0

- Consolidated training, validation-only threshold tuning, held-out evaluation,
  report generation, artifact persistence, and governance under one installed
  CLI.
- Preserved the official UCI test partition and jointly stratified validation
  by label, sex, and binary race.
- Added strict report parsing, package provenance, locked dependencies, real
  artifact API tests, and a reference run bundle.
