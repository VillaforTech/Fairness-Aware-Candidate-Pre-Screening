# Evaluation methodology

## 1. Declare the benchmark

The prediction target is the UCI Adult income label. The experiment uses a
global probability threshold as its baseline, sex as the default binary policy
attribute, and original UCI race for intersectional diagnostics.

The benchmark does not stand in for applicant quality or job performance. That
scope is recorded in every report.

## 2. Prepare and validate data

Raw train and test files are cleaned deterministically. The model-ready table
must pass exact schema, dtype, category, null, and range checks.

`fnlwgt` stays in the table for sampling-weight sensitivity but is excluded from
the feature matrix. Sex and race fields stay in the audit plane only.

Before deletion, the raw audit measures complete-case attrition, missingness,
and group composition. Both raw and processed audits record exact duplicates,
repeated 11-feature vectors, conflicting labels for the same feature vector,
cross-split feature-vector overlap, and small-group evidence flags. The default
rebuild removes 3,620 of 48,842 rows, a 7.41% complete-case attrition rate.

The `.quality.json` sidecar binds both raw source files and the model-ready CSV
by SHA-256. Experiment startup recomputes the processed audit, rejects a sidecar
that disagrees with the loaded CSV, embeds the evidence in the report, and
records the sidecar digest in report and manifest.

## 3. Preserve the test partition

The official UCI test rows remain fixed. Validation is drawn from the original
training file with joint stratification by `income`, `sex`, and `race_binary`.
Preprocessing and the classifier are fitted on fit rows only.

This design prevents threshold selection from using test labels and reduces the
chance that a random validation draw removes a small label/group cell.

## 4. Fit one probability model

The preprocessing and estimator live in one scikit-learn pipeline. The same
fitted pipeline produces validation and test probabilities. Model-specific
parameters and transformed feature names are recorded in the artifact.

The canonical numeric fields are `age`, `education_num`, `capital_gain`,
`capital_loss`, and `hours_per_week`. The canonical categorical fields are
`workclass`, `education`, `marital_status`, `occupation`, `relationship`, and
`native_country`. Their order is fixed independently of dataframe column order.
Numeric fields are standardized. Categorical fields are one-hot encoded from
the fitted training vocabulary with `handle_unknown="ignore"`.

Ignoring OOV values is limited to evaluation transformation: an unseen category
produces zeros for that field's one-hot block rather than stopping validation or
test evaluation. Before fitting, the run compares validation and test
categories with fit-row vocabularies and records split and per-column OOV values,
row counts, and shares. The simulation boundary is stricter and rejects an OOV
category before scoring.

The run uses a cloned configuration. The effective model type and validation
ratio are synchronized into `model.model_type` and `data.val_size`; the effective
seed is synchronized into `seed` and `model.random_state` and passed to the split
and estimator. The caller's configuration is not mutated. This effective
resolved configuration is serialized and SHA-256 hashed for report and manifest
evidence.

## 5. Select the offline frontier policy

For a configured threshold grid, the optimizer evaluates the Cartesian product
of privileged and unprivileged thresholds on validation data.

Each candidate records:

- accuracy and loss relative to the global threshold;
- privileged and unprivileged selection rates;
- SPD-equivalent signed selection-rate gap and DI;
- privileged and unprivileged TPR and signed TPR gap; and
- privileged and unprivileged FPR and signed FPR gap.

The Pareto objectives are maximum accuracy and minimum absolute TPR gap. A point
is removed when another candidate is at least as good on both objectives and
strictly better on one. Repeated operating points are collapsed
deterministically.

Selection then chooses the most accurate point satisfying maximum absolute TPR
gap and maximum accuracy loss. Ties prefer smaller absolute TPR gap, FPR gap,
selection-rate gap, and policy movement. If no candidate is feasible, the
global-threshold baseline is evaluated and the infeasible status is preserved.

The project also records its earlier one-sided opportunity comparator. It does
not control the selected v2 policy.

## 6. Select the global review band

The review selector considers symmetric half-widths around the global base
threshold. Each candidate measures validation automation coverage and error
among automated cases. The canonical experiment selects this policy without
Census weights.

Feasibility requires:

- automated error no greater than the configured maximum; and
- at least the configured number of automated validation rows.

The selected candidate maximizes coverage. The frozen band is then evaluated on
the held-out test rows. The report includes overall and intersectional review
burden, automation coverage, automated error, automated accuracy, and whether
the validation-selected error constraint still holds.

Review means abstention from the automatic simulation. It does not model a
reviewer's decision.

### Validation-overlap dependence

After the canonical selectors run, the audit compares validation rows with fit
rows by exact canonical 11-feature identity. It records train and validation
row counts, exact-feature overlap count and rate, exact full-record overlap count
and rate, and the number of validation rows left after feature-overlap exclusion.

When rows remain, the frontier and review-band selectors are rerun on only the
overlap-excluded validation rows. The model and validation probabilities remain
fixed. These alternate policy outputs are dependence evidence and do not replace
the canonical policies used on held-out test rows. If no rows remain or policy
selection cannot be estimated, the report records `not_estimable` and its
reason.

## 7. Evaluate paired held-out outcomes

Baseline and adjusted policies are applied to the same test probabilities.
Reported metrics include:

- accuracy, precision, recall, F1, ROC AUC, PR AUC, and Brier score;
- SPD and DI;
- group TPR, FPR, and their signed privileged-minus-unprivileged gaps; and
- label, prediction, and confusion counts for each observed audit group.

The probability metrics do not change between baseline and group-threshold
predictions because the underlying score model is the same. Decision metrics
can change because thresholds change.

## 8. Quantify test-sample uncertainty

The paired bootstrap resamples test rows within label by policy-group strata.
Each replicate evaluates both baseline and adjusted predictions on identical
sampled indices. The report stores percentile intervals for baseline, adjusted,
and change values of:

- accuracy;
- SPD;
- DI;
- TPR gap; and
- FPR gap.

Pairing preserves the comparison and stratification avoids empty policy-group
label cells created only by resampling. The method still conditions on the
observed test data and chosen model. It is not a Census design-based interval,
an external-validity estimate, or a guarantee under shift.

## 9. Audit intersections and calibration

Observed sex by original-race cells receive:

- raw and weighted support;
- positive and negative label counts;
- confusion counts;
- selection rate, TPR, and FPR;
- Brier score and fixed-width expected calibration error; and
- interval and evidence metadata.

Unweighted rates use Wilson score intervals. Weighted sensitivity uses Wilson
intervals with Kish effective sample size. Evidence state is:

- `sufficient` when configured support and class counts hold;
- `limited` when an estimate exists but support criteria fail; or
- `not_estimable` when a required denominator is zero.

Worst-group spans use only sufficient cells and report
`insufficient_groups` when fewer than two eligible groups remain. The raw cells
stay in the report either way.

## 10. Remove exact feature overlaps without retuning

The overlap audit builds canonical tuples from the ordered 11 model features.
Train and validation rows form the reference set because they contribute to
model fitting or policy selection. Test rows whose exact feature tuple appears
in that reference set are identified with sorted tuple lookup and final value
equality, not digest equality.

The report preserves metrics for the complete held-out set and recomputes them
on the overlap-excluded slice. The model, probabilities, baseline predictions,
adjusted predictions, and thresholds remain frozen. Labels, protected fields,
split markers, and `fnlwgt` are ignored for identity, so conflicting labels do
not hide a repeated feature vector.

This held-out sensitivity is not the validation-overlap retuning described in
step 6. No selector is rerun and no held-out label can change a policy.

Each slice records whether binary-group fairness metrics remain estimable. An
empty or denominator-deficient novel slice is explicit evidence, not a zero.
The sensitivity tests dependence on repeated benchmark identities; it does not
make the remaining observations independent or externally representative.

## 11. Compare sampling-weight sensitivity

The full held-out metric set is computed again with `fnlwgt` as a sample weight.
This answers whether weighted and unweighted descriptions differ materially. It
does not turn the benchmark into a current population estimate or provide a
complete survey-design analysis.

## 12. Apply the gate

The gate validates report structure, then checks performance loss, group gaps,
DI, SPD, interval bounds, adjusted intersectional spans, and held-out review
behavior. Every violation is retained. A favorable movement in one metric
cannot erase an unfavorable result elsewhere.

The gate persists its exact threshold set with the verdict. Rechecking a report
without explicit overrides reuses those persisted thresholds. Bundle save and
load both recompute the verdict and require it to match the copies in report and
manifest. They also require the serving review-band policy and offline
group-threshold policy to match their corresponding report selection evidence,
held-out thresholds, and protocol fields.

## 13. Repeat across seeds

The stability study reruns steps 3 through 12 for each seed. Aggregation requires
identical source, data, model type and parameters, resolved configuration after
removing only `seed` and `model.random_state`, protocol, feature contract, gate
threshold policy, and metric coverage.

It summarizes minimum, quartiles, median, maximum, mean, and population standard
deviation for supported metrics, threshold distributions, gate pass rate,
worst gap by metric, and a deterministic worst-seed ranking.

Because every run uses the same official test partition, variation reflects
training, split, and policy-selection sensitivity. It does not measure
independent population sampling uncertainty.

## 14. Publish evidence atomically

The run bundle is written only after model, policy, report, predictions, HTML,
aggregate monitoring reference, digests, and cross-document bindings validate.
Run-specific results should be cited from that seven-file bundle, not copied
from an unrelated or earlier execution.

Every writer documented as atomic uses a hidden temporary path in the final
destination's parent directory, a fresh UUID suffix, and `os.replace`. This
avoids a shared temporary filename across concurrent writers. The rule covers
preprocessing CSV and JSON, general JSON and monitoring outputs, batch output,
bundle publication, and incomplete-study markers.

## 15. Compare offline drift evidence

The run's `monitoring.json` is built from official held-out features, baseline
scores and predictions, audit groups, delayed labels, and audit-only sample
weights. It stores aggregate distributions and 101-point quantile sketches, not
rows.

A current snapshot must pass the same strict role and dtype validation. The
comparison measures numeric PSI and quantile-based KS-like distance,
categorical total variation and OOV share, score and selection drift, group
composition, and optional delayed-label performance and group-gap drift.

Validation also checks whether serialized aggregates agree with one another.
Category counts must produce their shares and unknown summaries; prediction
counts must produce selection rate; label counts, confusion totals, and
protected-group totals must agree; and binary rates must be derivable from their
confusion denominators. A structurally valid but internally contradictory
snapshot is rejected before comparison.

Violations produce `FAIL`. Evidence gaps without a violation produce
`INSUFFICIENT_EVIDENCE`. Only a comparison with no violations or required gaps
produces `PASS`. These are descriptive offline thresholds with no p-values, not
evidence that an operating context is valid or safe.
