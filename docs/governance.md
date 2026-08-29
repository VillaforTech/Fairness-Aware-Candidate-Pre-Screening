# Experimental policy gate

The gate is a strict parser plus a set of configurable experiment checks. It is
not a deployment approval, legal test, or ethical certification.

## Default checks

| Check | Default |
|---|---:|
| Adjusted accuracy | at least 0.80 |
| Accuracy drop from baseline | at most 0.02 |
| Absolute TPR gap | at most 0.05 |
| Disparate impact | 0.80 to 1.25 |
| Absolute SPD | at most 0.10 |

The DI interval is two-sided so reverse disparity is not treated as unlimited
success. Accuracy loss is computed from baseline and adjusted metrics in the
same report.

## Fail-closed report contract

Schema `1.0` requires:

- a nonempty string `run_id`
- a non-Boolean, nonnegative integer `seed`
- `model_type` equal to `lr`, `rf`, or `xgb`
- a full lowercase 40-hex Git commit and Boolean dirty state when executed from
  a checkout, or the explicit pair `"unavailable"` / `null` for an installed
  distribution
- 64-hex SHA-256 fingerprints for the input data and canonical package source
- baseline accuracy
- adjusted accuracy, TPR gap, DI, and SPD

Required metrics must be finite numbers in their possible domains. The gate
rejects `null`, strings, Booleans, NaN, infinity, impossible values, missing
objects, and unsupported schema versions.

```json
{
  "schema_version": "1.0",
  "metadata": {
    "run_id": "xgb-seed-42",
    "seed": 42,
    "model_type": "xgb",
    "git_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "dirty_worktree": false,
    "data_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "source_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  },
  "results": {
    "baseline_metrics": {
      "accuracy": 0.8694
    },
    "metrics": {
      "accuracy": 0.8678,
      "TPR_gap": -0.0124,
      "DI": 0.4120,
      "SPD": 0.1563
    }
  }
}
```

The canonical `fairness audit` command writes the complete schema, including
protocol, group cells, thresholds, uncertainty, and the recorded verdict.
The source fingerprint hashes the installed `fairness_project` package itself,
so it is nonempty and stable both inside a checkout and from a wheel. Git
commands are anchored to that package's owning checkout and never run against
the caller's current directory.

## Commands and exit codes

```bash
uv run fairness gate --report runs/xgb-seed-42/report.json
```

- `0`: valid report, configured checks passed
- `1`: valid report, at least one configured check failed
- `2`: unreadable/malformed report or invalid threshold configuration

Structural validation happens before policy evaluation. Consequently, a bad
digest, unsupported model type, Boolean seed, missing metric, or impossible
metric value is an input error (`2`), not evidence that a valid model report
missed a policy threshold (`1`). Programmatic callers can distinguish the two
using `GateResult.report_valid` or `GateResult.exit_code` while continuing to
use `check_gate(report)` as before.

Advanced threshold overrides remain available on the module CLI:

```bash
uv run python -m fairness_project.governance.gate \
  --report runs/xgb-seed-42/report.json \
  --min-accuracy 0.85 \
  --max-accuracy-drop 0.01 \
  --max-tpr-gap 0.03 \
  --min-di 0.85 \
  --max-di 1.18 \
  --max-spd 0.08
```

An override changes an experiment criterion. It does not make a failing model
safe or compliant.

## CI behavior

CI rebuilds the model-ready data from the bundled UCI files, trains Logistic
Regression through the same leakage-free path, generates a real report, and
requires the default gate to reject its DI/SPD result. This tests report
generation and policy wiring together; fixed unit fixtures separately exercise
pass, fail, malformed, and non-finite cases.

## Interpretation

The reference XGBoost run improves TPR gap, DI, and SPD while losing 0.0016
accuracy. It still fails DI and SPD. That is the intended outcome of the gate:
preserve the whole trade-off and block a one-metric success story.

Passing would mean only that these configured numeric checks passed on this
dataset and split. It would not establish construct validity, external validity,
job relatedness, legality, privacy, security, or accountable operation.
