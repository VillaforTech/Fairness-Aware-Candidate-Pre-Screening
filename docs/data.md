# Data documentation

## Source

The project uses the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult),
derived from 1994 US Census data and donated by Barry Becker and Ronny Kohavi.
UCI lists the dataset under CC BY 4.0.

Suggested citation:

```text
Becker, B. & Kohavi, R. (1996). Adult [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
```

The raw distribution contains 48,842 rows: 32,561 in `adult.data` and 16,281
in `adult.test`. Rows containing UCI's `?` missing-value marker are removed,
leaving 45,222 model-ready rows in the current rebuild.

## Columns and model contract

| Column | Role | Model feature |
|---|---|---:|
| `age` | numeric attribute | yes |
| `workclass` | categorical attribute | yes |
| `fnlwgt` | census sampling weight | yes, with caveat |
| `education` | categorical attribute | yes |
| `education_num` | numeric education encoding | yes |
| `marital_status` | categorical attribute | yes |
| `occupation` | categorical attribute | yes |
| `relationship` | categorical attribute | yes |
| `native_country` | categorical attribute | yes |
| `capital_gain` | numeric attribute | yes |
| `capital_loss` | numeric attribute | yes |
| `hours_per_week` | numeric attribute | yes |
| `sex` | protected attribute for offline audit | no |
| `race` | protected attribute for offline audit | no |
| `race_binary` | derived White/Non-White audit grouping | no |
| `income` | target, `>50K` versus `<=50K` | no |
| `split` | official partition marker | no |

`fnlwgt` is a census sampling weight, not an ordinary personal characteristic.
The reference preserves the original prototype feature choice but flags it
for a future weighted-sensitivity analysis rather than presenting that choice
as settled.

## Rebuild and derived files

Raw UCI files are kept under `data/raw/adult/`. Cleaned CSVs, predictions, and
model binaries are derived artifacts and are intentionally not versioned.

```bash
uv run fairness data preprocess \
  --input-dir data/raw/adult \
  --output-path data/processed/adult/adult_model_ready.csv
```

The command strips whitespace, standardizes test labels, removes missing rows,
preserves the original train/test markers, and derives `race_binary`.

## Split protocol

The external UCI test partition is never resampled. With seed 42, 15% of the
cleaned original training partition is assigned to validation using joint
stratification over `income`, `sex`, and `race_binary`.

| Split | Rows | Purpose |
|---|---:|---|
| Fit | 25,637 | fit preprocessing and classifier |
| Validation | 4,525 | tune the offline threshold policy |
| Test | 15,060 | one final held-out evaluation |

The splitter validates the ratio and required columns, rejects sparse strata
that cannot be divided safely, does not mutate NumPy's global random state, and
records every split/group/label cell in the report.

## Feature preprocessing

Numeric features are standardized and categorical features are one-hot encoded
inside the fitted scikit-learn pipeline. Unknown categories at inference use
`OneHotEncoder(handle_unknown="ignore")`. Missing, null, or extra API/batch
columns are rejected before the model is called.

## Lineage and provenance

```text
Bundled UCI raw files
  -> deterministic cleaning and split labels
  -> SHA-256 fingerprinted model-ready CSV
  -> joint-stratified fit/validation split; official test preserved
  -> fitted 12-feature preprocessing + classifier
  -> validation-only threshold tuning
  -> frozen test evaluation and paired bootstrap
  -> model + manifest + policy + report + predictions bundle
  -> baseline-only API and batch inference
```

Each report records the data hash, canonical package-source hash, seed, model
type, Python and dependency versions, split cells, thresholds, subgroup
diagnostics, uncertainty, and policy verdict. Source checkouts also record the
owning checkout's full Git commit and dirty-worktree flag; installed
distributions use an explicit unavailable/null pair and never inspect the
caller's repository. The manifest also records model, report, and policy
digests; loading fails closed when the bundle is incomplete, corrupted, or no
longer matches those recorded digests. Because the manifest is not signed, this
is an integrity check rather than proof against an attacker who can rewrite the
entire bundle.

## Data limitations

- Adult is income data, not applicant, qualification, performance, or hiring
  data.
- The records are historical and encode social and economic inequities.
- Removing missing rows can change group representation and is not neutral.
- Binary group mappings erase identity, intersectionality, and uncertainty.
- The label is imbalanced and some intersectional positive-label cells are
  small.
- The reference is unweighted despite `fnlwgt`; this limits population-level
  interpretation.

These data support a benchmark fairness audit only. They do not support
claims about real candidates or employment decisions.
