# Offline monitoring evidence

The monitoring module builds and compares strict aggregate snapshots. It is a
replayable drift-audit tool for offline data, not a live service, alerting
system, or production approval mechanism.

Every experiment bundle includes `monitoring.json`. The reference is built from
the official held-out rows with:

- all 11 scoring features;
- baseline model score and prediction;
- aggregate sex, original-race, and `race_binary` composition;
- delayed benchmark label metrics; and
- `fnlwgt` audit-only sensitivity.

The artifact is validated before publication and bound by
`manifest.monitoring_sha256`.

## Snapshot contract

A snapshot contains aggregate statistics only:

- numeric count, mean, standard deviation, extrema, and a 101-point quantile
  sketch;
- categorical counts, shares, and unknown-token share;
- score distribution and prediction selection rate;
- protected-column counts and shares;
- optional delayed-label performance and per-group metrics; and
- optional sample-weight summary and weighted sensitivity.

It never contains input rows or row-level protected data. Aggregate-only does
not mean anonymous: rare category counts and small groups can still be
sensitive. Review a snapshot before sharing it.

Declared columns must exactly cover the input CSV. Feature, categorical,
protected, score, prediction, label, and weight roles must be distinct.
Protected fields and the sample weight cannot be model features. Numeric values
must be finite, scores must lie in `[0, 1]`, and predictions and labels must be
binary.

Validation checks derived aggregates, not only field names and numeric domains:

- categorical counts must sum to row count, shares must match counts, and
  unknown counts and shares must match the fixed unknown-token set;
- prediction counts must sum to row count and produce the stored selection
  rate;
- delayed-label class counts must match overall class and confusion totals;
- protected-group counts and confusion totals must roll up to their categorical
  summaries and overall delayed-label metrics;
- accuracy, TPR, FPR, precision, base rate, and selection rate must match their
  confusion denominators, including explicit `null` for a zero denominator; and
- weighted performance must satisfy the same internal confusion and rate
  equations, while weighted group-composition shares must sum to one.

A snapshot with plausible but contradictory aggregates is rejected before
bundle publication, bundle load, or comparison.

## Create a current snapshot

The following command matches the reference roles produced by the canonical
Adult experiment:

```bash
uv run fairness monitor snapshot \
  --input-csv offline/current.csv \
  --output-json offline/current.json \
  --feature-columns age,workclass,education,education_num,marital_status,occupation,relationship,native_country,capital_gain,capital_loss,hours_per_week \
  --categorical-columns workclass,education,marital_status,occupation,relationship,native_country \
  --score-column score \
  --prediction-column prediction \
  --protected-columns sex,race,race_binary \
  --label-column label \
  --sample-weight-column fnlwgt
```

The CSV must contain exactly the declared columns. Omit both the optional column
and its CLI argument when delayed labels or sample weights are unavailable.

Snapshot JSON is written atomically through a hidden sibling temporary path with
a fresh UUID suffix and `os.replace`. Comparison JSON uses the same writer, so
concurrent writers do not contend for one fixed temporary filename.

## Compare with a run reference

```bash
uv run fairness monitor compare \
  --reference-json runs/xgb-seed-42/monitoring.json \
  --current-json offline/current.json \
  --output-json offline/comparison.json \
  --require-pass
```

The comparison validates both snapshots before calculating drift. Feature,
categorical, numeric, score, prediction, and protected roles must match.
Required-column dtypes must match as well. Incompatible contracts are errors,
not drift results.

`--require-pass` exits `1` for both `FAIL` and `INSUFFICIENT_EVIDENCE`. Without
that flag, the command writes the comparison and prints its status without using
the status as the process exit code.

## Compared evidence

| Surface | Measures |
|---|---|
| Numeric features | Population Stability Index and KS-like distance reconstructed from quantile sketches |
| Categorical features | Total variation, categories outside the reference vocabulary, OOV share, and unknown-share increase |
| Model score | Population Stability Index, KS-like distance, and mean change |
| Prediction | Selection-rate change |
| Protected groups | Aggregate composition total variation and support |
| Delayed labels | Accuracy, TPR, FPR, precision, base rate, and selection-rate change |
| Delayed-label group audit | Change in selection-rate, TPR, and FPR spans for eligible observed groups |
| Sampling weight | Audit status and weighted sensitivity, excluded from the primary drift gate |

The numeric distance is descriptive and reconstructed from aggregate quantiles.
The comparison emits no hypothesis-test p-values.

## Default drift policy

| Check | Default |
|---|---:|
| Minimum rows per snapshot | 100 |
| Minimum rows per protected group | 30 |
| Numeric PSI | at most 0.25 |
| Numeric KS-like distance | at most 0.20 |
| Categorical total variation | at most 0.20 |
| Categorical OOV share | at most 0.05 |
| Unknown-share increase | at most 0.05 |
| Score PSI | at most 0.25 |
| Score KS-like distance | at most 0.20 |
| Absolute selection-rate change | at most 0.10 |
| Protected-group composition total variation | at most 0.10 |
| Delayed-label accuracy drop | at most 0.05 |
| Delayed-label TPR drop | at most 0.10 |
| Delayed-label FPR increase | at most 0.10 |
| Selection-rate group-span increase | at most 0.10 |
| TPR group-span increase | at most 0.10 |
| FPR group-span increase | at most 0.10 |
| Delayed labels required | no |

These are repository policy defaults. They are not statistical critical values,
service-level objectives, or legal thresholds.

## Status semantics

### `PASS`

No configured threshold was violated and no required evidence gap was found.
If delayed labels are absent under the default policy, label and fairness drift
are skipped. A pass then applies only to the evidence that was available.

### `FAIL`

At least one configured threshold was violated. Violations take precedence even
when the comparison also contains evidence gaps.

### `INSUFFICIENT_EVIDENCE`

No threshold violation was detected, but minimum rows, protected-group support,
required labels, or metric estimability did not hold. The gate is fail-closed:
insufficient evidence is never reported as a pass.

## Operational interpretation

The run reference is derived from the same historical Adult held-out partition
used for evaluation. Comparing a new snapshot to it can expose data-contract
breaks and descriptive shifts. It cannot establish that the reference is
representative of a current population or suitable as an operating baseline.

The module does not schedule collection, ingest live traffic, page an owner,
retain an incident history, identify root cause, stop an external system, or
validate an employment use case. Those responsibilities require an accountable
operating process outside this repository.
