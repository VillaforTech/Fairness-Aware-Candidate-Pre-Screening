# Interpretation contract

## Purpose

This repository is an auditable policy lab built around UCI Adult. It studies
how an income-classification decision policy changes measured utility, group
disparities, uncertainty, and review burden under a fixed benchmark protocol.

The useful output is the evidence chain:

```text
raw data semantics -> split discipline -> policy search -> held-out metrics
                   -> uncertainty -> subgroup evidence -> stability -> gate
                   -> aggregate drift reference -> offline comparison
```

No single fairness metric closes that chain.

## Questions the experiment can answer

Within one declared configuration, the lab can show:

- how complete-case deletion changes support and observed group composition;
- where feature-vector collisions, conflicting labels, and cross-split overlap
  limit a simple benchmark interpretation;
- which group-threshold pairs were considered on validation data;
- which pairs are nondominated in accuracy versus absolute TPR gap;
- why one feasible point was selected, or why none was feasible;
- how the frozen policy changed held-out performance and group rates;
- how paired bootstrap intervals change the interpretation of point estimates;
- where observed sex by race cells have sufficient, limited, or non-estimable
  evidence;
- how `fnlwgt`-weighted rates differ from unweighted rates;
- how a global review band trades automation coverage for automated error;
- whether validation-selected review behavior holds on the test partition;
- whether a comparable offline snapshot passes, fails, or lacks sufficient
  evidence under the configured drift policy; and
- how metrics, thresholds, and gate outcomes vary across repeated seeds.

These are benchmark and engineering questions. They do not establish whether a
real decision process should exist.

## Policy separation

The repository evaluates two mechanisms with different information needs.

### Offline group thresholds

The Pareto search uses validation labels and one binary protected-group field.
That access is explicit and confined to offline evaluation. The selected policy
is never exposed through the API.

This matters because group-specific cutoffs in a real employment process may be
unlawful or inappropriate. The project does not treat a smaller TPR gap as
permission to use such a policy.

### Global review band

The local simulator uses one group-blind probability band. It can abstain from
an automatic outcome, but it does not implement human review. The report treats
review as burden and unresolved work, not as a guaranteed correction.

Human decision makers can introduce inconsistency, delay, bias, privacy risk,
and automation deference. Those outcomes are absent from Adult and cannot be
inferred from review coverage.

## Measurement discipline

### Data semantics precede model metrics

The preprocessing sidecar records 7.41% complete-case attrition and preserves
group-specific missingness and composition evidence. It also exposes repeated
11-feature vectors, label conflicts, and train/test feature overlap. These are
properties of benchmark rows and feature identities, not proof that two rows
belong to one person.

The experiment verifies the sidecar against the model-ready CSV, recomputes the
processed audit, embeds the evidence, and binds the sidecar digest. This keeps
data limitations attached to model results.

It also evaluates the frozen predictions again after excluding held-out rows
whose exact 11-feature identity appeared in train or validation. That view can
reveal sensitivity to repeated benchmark records, but it cannot make the
remaining slice representative, independent, or suitable for employment use.

### Multiple views, no single score

The audit keeps accuracy, TPR gap, FPR gap, SPD, DI, calibration, subgroup
support, and review burden together. Improvements can conflict. A change that
narrows TPR gap can widen FPR gap, alter selection rates, or move burden toward
one intersectional group.

### Uncertainty is part of the result

The paired bootstrap resamples the same held-out rows for baseline and adjusted
policies while preserving label and policy-group cells. It estimates
test-sample variability for that paired comparison. It does not cover a new
population, a different dataset, model selection, the Census sample design, or
unknown future shift.

Intersectional rates use Wilson intervals. Weighted sensitivity uses Kish
effective sample size and is labelled accordingly. Support thresholds change an
evidence state, not the underlying observed count.

### Stability is separate from sampling uncertainty

Repeated-seed studies retrain and retune the system. They reveal sensitivity to
the fit/validation split, estimator randomness, and policy selection. The runs
share the official test partition and therefore are not independent test
samples.

### Offline drift is separate from both

The monitoring reference stores aggregate held-out distributions and quantile
sketches. A comparison can flag feature, score, selection, group-composition,
and optional delayed-label shifts. It emits no p-values and does not identify a
cause. `INSUFFICIENT_EVIDENCE` is distinct from `PASS`, while any configured
violation produces `FAIL`.

## Governance behavior

The gate checks strict structure before policy criteria. A malformed report is
an error, a valid threshold miss is a rejection, and only a report satisfying
every configured check passes.

Interval bounds, intersectional spans, and held-out review behavior can reject
a run even when headline point metrics look acceptable. This is deliberate.
Evidence should fail closed when its own uncertainty or weakest observed group
contradicts the preferred story.

The thresholds are repository policy, not universal definitions of fairness.
They must not be presented as legal safe harbors.

The offline drift gate is a second, independent evidence check. A policy-gate
pass cannot override drift failure, and a drift pass cannot validate the model,
dataset, or decision context.

## Employment boundary

Adult predicts an income category from historical Census-derived attributes.
It provides no basis for claims about:

- job analysis or essential functions;
- applicant qualifications;
- predictive validity for work performance;
- business necessity;
- accommodations or disability-related testing;
- representative applicant flows;
- candidate consent, notice, or privacy expectations;
- the reliability of a human review process;
- appeal, correction, and adverse-action procedures; or
- outcomes after a decision.

US employment guidance requires attention to the specific procedure, job, and
validation evidence. Other jurisdictions add different requirements. See the
primary sources in [`references.md`](references.md). Repository metrics cannot
substitute for domain, legal, and affected-stakeholder review.

## Misuse controls in the implementation

- The feature contract excludes sex, race, `race_binary`, and `fnlwgt`.
- The API rejects those fields and all extras.
- The API uses the verb `simulate` and labels itself evaluation-only.
- Group thresholds are persisted as offline-only and are not served.
- Every response identifies the artifact and policy.
- Rejected artifacts require an explicit research override.
- Unknown categories fail instead of silently using an all-zero encoding.
- Every bundle is atomically written and hash-bound before loading.
- Raw and processed data-semantics evidence is verified and bound to each
  canonical run.
- Exact-overlap sensitivity keeps model scores and policy choices frozen and
  preserves non-estimable novel-only evidence.
- Every bundle contains a validated, hash-bound, aggregate-only monitoring
  reference.
- The generated report carries the gate verdict and its violations.

These controls reduce accidental overstatement and policy confusion. They do
not prevent a determined person from repurposing the code.

## Requirements before any real high-impact use

A real project would begin again with the intended context. At minimum it would
need a defensible target, job-specific validation, representative prospective
data, independent legal and domain review, affected-stakeholder participation,
privacy and security engineering, accessibility and accommodations, monitored
human decision quality, appeal and contestability, drift ownership, incident
response, and an accountable authority able to stop the system.

Those requirements cannot be satisfied by improving Adult benchmark metrics.
