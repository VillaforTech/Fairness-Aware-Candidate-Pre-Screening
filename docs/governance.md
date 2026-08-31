# Evidence gate

The gate is a strict report parser followed by configurable policy checks. It
answers one narrow question: does this completed audit satisfy every numeric
criterion encoded for this repository?

A pass is not a deployment approval, legal opinion, ethical certification, or
claim of validity for employment.

## Default checks

| Check | Default |
|---|---:|
| Adjusted accuracy | at least 0.80 |
| Accuracy drop from baseline | at most 0.02 |
| Absolute TPR gap | at most 0.05 |
| Absolute FPR gap | at most 0.05 |
| Disparate impact | 0.80 to 1.25 |
| Absolute SPD | at most 0.10 |
| Intersectional TPR span | at most 0.10 |
| Intersectional FPR span | at most 0.10 |

If a paired bootstrap section is present, the worst absolute interval bound for
TPR gap, FPR gap, and SPD must also remain within its limit. The DI interval
must remain inside both configured bounds.

If a selective-review section is present, a validation-selected review band
must still meet its automated-error constraint on held-out data. The gate also
checks adjusted intersectional TPR and FPR spans when those spans are estimable.

The defaults are explicit engineering criteria. The DI range is not a shortcut
for statistical significance, practical importance, job relatedness, or
jurisdiction-specific legal analysis.

The verdict contains the exact nine threshold values used for that evaluation.
When `check_gate` or the gate CLI rechecks a completed report without explicit
overrides, it reconstructs the threshold policy from
`report.governance.thresholds`. Missing persisted governance falls back to the
defaults for a report without a governance block. A malformed or partial
persisted threshold object is an error, not permission to substitute defaults.

## Fail-closed schema

Report schema `2.0` requires:

- a nonempty `run_id`;
- a non-Boolean, nonnegative seed;
- model type `lr`, `rf`, or `xgb`;
- a full lowercase 40-hex Git commit and Boolean dirty state when the source is
  a checkout, or the explicit `unavailable` and `null` pair for an installed
  distribution;
- 64-hex SHA-256 fingerprints for model-ready data and package source;
- baseline accuracy; and
- adjusted accuracy, TPR gap, FPR gap, DI, and SPD.

Required metrics must be finite and inside their mathematical domains. Missing
objects, strings in numeric fields, Booleans, NaN, infinity, impossible values,
and unsupported schemas make the report invalid. An invalid report is not
treated as evidence that a valid model merely missed a threshold.

The canonical audit report also carries:

- resolved configuration and its digest;
- synchronized effective model type, validation ratio, seed, and estimator
  `random_state` inside that resolved configuration;
- verified raw and processed data-semantics evidence plus the sidecar digest
  when present;
- model parameters and preprocessing contract;
- split-level validation and test OOV evidence against the training vocabulary;
- fit, validation, and official-test cells;
- the full validation Pareto frontier and selected point;
- a one-sided comparator retained for analysis;
- frozen held-out thresholds;
- weighted and unweighted metrics;
- subgroup and intersectional diagnostics;
- paired bootstrap intervals;
- exact-feature overlap counts and frozen-policy metrics for the complete and
  overlap-excluded held-out slices;
- train-to-validation feature and full-record overlap counts plus
  overlap-excluded validation retuning evidence;
- selective-review validation and held-out results;
- an aggregate-only monitoring-reference summary; and
- the persisted gate verdict.

## Commands and exit codes

```bash
uv run fairness gate --report runs/xgb-seed-42/report.json
```

| Exit | Meaning |
|---:|---|
| `0` | Valid report and every configured check passed |
| `1` | Valid report and at least one configured check failed |
| `2` | Report or gate configuration was malformed |

The training command saves the verdict without failing by default, because a
well-formed rejection is a useful experiment result. Use
`fairness audit --require-gate-pass` when a calling workflow should stop on exit
`1`.

Advanced overrides are available through the module entry point:

```bash
uv run python -m fairness_project.governance.gate \
  --report runs/xgb-seed-42/report.json \
  --min-accuracy 0.85 \
  --max-accuracy-drop 0.01 \
  --max-tpr-gap 0.03 \
  --max-fpr-gap 0.03 \
  --max-intersectional-tpr-span 0.08 \
  --max-intersectional-fpr-span 0.08 \
  --min-di 0.85 \
  --max-di 1.18 \
  --max-spd 0.08
```

Changing a criterion changes only the experiment policy. It does not change
the data's meaning or establish that a result is safe. A command-line override
applies to that gate invocation; it does not rewrite the report file.

## Artifact behavior

The verdict, including its thresholds, is written to both `report.json` and
`manifest.json`. Bundle save and bundle load recompute the gate from report
evidence using the persisted threshold set, then require all three verdicts to
match. A digest-rehashed report with stale metric checks or a stale pass flag is
therefore rejected. The local simulation service refuses a rejected artifact
unless the caller supplies an explicit research override. This prevents a
rejection from becoming a silent default.

The manifest binds the report by SHA-256 and the loader verifies report and
manifest agreement. It also binds the data-quality digest across report and
manifest and the complete `monitoring.json` file by SHA-256. Because the
manifest is not signed, the mechanism detects accidental or partial mutation,
not a malicious rewrite of the entire bundle.

The policy file is also evidence-bound, not merely hash-bound. The loader
requires the global serving policy ID, kind, thresholds, selection split,
review label, error limit, and protected-attribute flags to match the report's
selective-review policy. The offline policy ID, kind, selection status,
group thresholds, group definitions, and tuning split must match validation and
protocol evidence. The offline group policy remains outside the API path.

## Separate offline drift gate

`fairness monitor compare` applies a different fail-closed policy to two
aggregate snapshots. Its status values are:

- `PASS`: no configured drift violation or required evidence gap;
- `FAIL`: at least one drift threshold violation; and
- `INSUFFICIENT_EVIDENCE`: no detected violation, but support, label, or
  estimability requirements were not met.

Violations take precedence over evidence gaps. `--require-pass` exits nonzero
for both non-pass statuses. The model-policy gate and offline drift gate do not
override one another. See [`monitoring.md`](monitoring.md).

## Reading a rejection

A rejected run can still be technically successful. It means the pipeline
produced valid evidence and at least one configured bound did not hold. The
correct response is to inspect the trade-off, intervals, cell support, review
burden, overlap-excluded sensitivity, and stability results. The overlap view
is descriptive and is not itself a gate rule. This includes both the
overlap-excluded validation retuning evidence and the separate fixed-policy
held-out overlap sensitivity. It is not appropriate to report only the metric
that moved in the preferred direction.

A passing run would remain bounded to this dataset, protocol, model, and policy
configuration. It would not establish construct validity, external validity,
causal fairness, job relatedness, legality, privacy, security, or accountable
operation.
