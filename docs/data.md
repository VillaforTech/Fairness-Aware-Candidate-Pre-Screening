# Data contract

## Source and meaning

The project uses the
[UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult), derived from
1994 US Census data. Its target is whether annual income exceeds USD 50,000.
UCI lists 48,842 rows, an original training file and test file, and a CC BY 4.0
license for the dataset.

Adult is not applicant data. It has no qualifications, applications,
interviews, hiring decisions, job roles, work performance, accommodations,
appeals, or downstream employment outcomes. It can exercise an audit pipeline,
but it cannot validate a selection procedure for work.

Suggested citation:

```text
Becker, B. & Kohavi, R. (1996). Adult [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
```

## Deterministic preparation

```bash
uv run fairness data preprocess \
  --input-dir data/raw/adult \
  --output-path data/processed/adult/adult_model_ready.csv
```

The command:

1. reads the bundled `adult.data` and `adult.test` files;
2. removes the first test-file comment row;
3. converts UCI's `?` markers to missing values;
4. removes rows with any missing value;
5. strips categorical whitespace and normalizes test labels;
6. preserves the source partition as `train` or `test`;
7. derives `race_binary` for a limited secondary audit view;
8. writes the model-ready CSV atomically; and
9. writes a deterministic `.quality.json` evidence sidecar atomically.

Both atomic writes use hidden sibling temporary files with fresh UUID suffixes,
then replace the destination. Concurrent writers do not share one predictable
temporary filename, and a failed write cleans up its own temporary path.

The current deterministic rebuild contains 45,222 complete rows. Removing
missing rows is itself a selection step. Group-specific attrition must be
examined rather than treated as harmless cleaning.

## Data-semantics evidence

The default output pair is:

```text
data/processed/adult/adult_model_ready.csv
data/processed/adult/adult_model_ready.quality.json
```

The sidecar binds the raw train file, raw test file, and model-ready CSV by
SHA-256. It contains separate raw and processed audits with:

- overall, split, sex, original-race, and sex by race complete-case attrition;
- missingness by column and observed group;
- before and after group composition;
- exact duplicate rows and repeated 11-feature vectors;
- repeated 11-feature vectors with conflicting income labels;
- overlapping 11-feature vectors across train and test, including label
  conflicts;
- small-group evidence flags; and
- explicit `fnlwgt` role and validity checks.

The canonical raw evidence reports:

| Check | Observed evidence |
|---|---:|
| Input rows | 48,842 |
| Complete-case rows | 45,222 |
| Rows removed | 3,620 |
| Complete-case attrition | 7.41% |
| Exact duplicate groups | 49 groups, 101 rows |
| Repeated 11-feature vectors | 4,369 groups, 13,975 rows |
| Conflicting-label 11-feature vectors | 1,127 groups, 4,352 rows |
| Train/test overlapping 11-feature vectors | 2,363 groups |
| Overlap groups with conflicting labels | 704 groups |

The 11-feature vector is a predictive feature identity, not a person identifier.
Equality does not prove that two rows describe the same person. The collision
and overlap counts are evidence about benchmark ambiguity and evaluation
dependence, not claims of duplicate individuals.

Each experiment separately uses the combined train and validation feature
identities as a reference, marks held-out overlap positions, and recomputes
baseline and adjusted metrics on the remaining rows without changing the model
or policy. This run-specific sensitivity is stored in `report.json`; the table
above remains the dataset-level lineage audit.

At experiment start, the processed audit is recomputed from the loaded CSV. If
the default sidecar exists, its model-ready digest and processed audit must
match exactly. The report embeds the raw and processed data-semantics evidence,
and the manifest records the sidecar digest as `data_quality_sha256`. A missing
sidecar remains explicit: the experiment can embed a fresh processed audit, but
raw attrition and the sidecar digest are unavailable.

## Three data planes

The lab separates fields by purpose.

### Scoring plane

The exact contract ID is `adult-income-v2-no-census-weight`.

| Feature | Type |
|---|---|
| `age` | integer |
| `workclass` | categorical |
| `education` | categorical |
| `education_num` | integer |
| `marital_status` | categorical |
| `occupation` | categorical |
| `relationship` | categorical |
| `native_country` | categorical |
| `capital_gain` | integer |
| `capital_loss` | integer |
| `hours_per_week` | integer |

The five numeric transformer inputs, in order, are `age`, `education_num`,
`capital_gain`, `capital_loss`, and `hours_per_week`. The six categorical
transformer inputs, in order, are `workclass`, `education`, `marital_status`,
`occupation`, `relationship`, and `native_country`. The fitted pipeline
standardizes the numeric group and one-hot encodes the categorical group. These
lists are canonical, not inferred from arbitrary model-ready column order.

The artifact records original and transformed feature names. The loader checks
the model input order, both transformer column lists, the encoder's unknown-value
mode, the fitted category arrays, and exact agreement between fitted transformed
names and manifest evidence.

The fitted category arrays are serialized inside the model file covered by
`model_sha256`. Their transformed names are copied into report and manifest
preprocessing evidence. The simulator reads its allowlist from those same fitted
arrays, so evaluation and serving cannot silently use independent vocabularies.

The training-fitted encoder uses `handle_unknown="ignore"`. During validation
and test evaluation, a category outside the training vocabulary is represented
by zeros in that field's one-hot block so the split remains evaluable. The run
does not hide this behavior: for both validation and test, it records rows with
any OOV value and, per categorical field, training-vocabulary size, distinct
unknown values, affected rows, and affected share.

The simulation service adds a stricter boundary. It extracts the category
vocabulary from the same fitted encoder, strips and validates incoming strings,
and rejects any category not observed during training before calling the model.

### Audit plane

| Field | Role |
|---|---|
| `sex` | primary binary group policy and audit attribute in the benchmark |
| `race` | original UCI category used for observed intersectional diagnostics |
| `race_binary` | derived White / Non-White grouping used for split balance and secondary diagnostics |
| `income` | binary benchmark target |
| `split` | official train/test marker, later extended with validation rows |

Protected attributes are retained so disparities can be measured. They are
excluded from the 11 model features and rejected by the simulation API.

The data's binary `sex` encoding and coarse race categories are historical
dataset fields. They do not represent the full range of identities. The derived
binary race field loses still more information, so the main intersectional
diagnostics use sex by original UCI race and carry explicit support limits.

### Weighting plane

`fnlwgt` is the Census final sample weight described by the source data
documentation. It is not an ordinary personal characteristic and is not a
model feature.

The audit uses `fnlwgt` only to compare weighted and unweighted held-out rates.
Weighted subgroup intervals use Kish effective sample size and are labelled as
sensitivity estimates. They are not Census design-based variance estimates,
because the benchmark does not provide or model the full sample design needed
for that claim.

## Split protocol

The official UCI test partition remains untouched. Validation rows are drawn
only from the original training partition, jointly stratified by:

- `income`
- `sex`
- `race_binary`

The splitter rejects invalid ratios, absent partitions, pre-existing validation
rows, missing stratification fields, and strata too small to divide. Each run
records split counts and every label/group cell.

```text
original UCI train
  |-- fit: preprocessing and classifier fitting
  `-- validation: threshold frontier and review-band selection

original UCI test
  `-- one paired evaluation of frozen policies
```

Test labels are used for evaluation, never for choosing thresholds or review
width.

Validation dependence is measured separately. Exact canonical feature identity
and exact full-record overlap are counted between fit and validation. Both
policy selectors are then rerun on validation rows whose 11-feature identity is
absent from fit, while the model and validation probabilities stay fixed. This
retuning is reported as sensitivity evidence and does not replace the policies
used on test.

Held-out overlap sensitivity uses train plus validation as its reference and
test as its compared slice. It does not retune. It only recomputes metrics after
filtering already frozen probabilities, predictions, and policy outputs.

## Lineage and integrity

Each report records:

- the model-ready CSV SHA-256;
- the data-quality sidecar SHA-256 when the canonical sidecar is present;
- embedded raw and processed data-semantics evidence;
- the package-source SHA-256;
- Git revision state when available;
- the resolved configuration and its SHA-256;
- the synchronized effective model type, validation ratio, seed, and model
  `random_state` in that resolved configuration;
- model parameters and dependency versions;
- the feature contract, fitted preprocessing vocabulary structure, transformed
  names, and validation/test OOV evidence;
- split seed, counts, and cells;
- policy thresholds and the selection protocol;
- weighted, unweighted, validation-overlap retuning, and fixed-policy held-out
  overlap diagnostic scope;
- a summary of the aggregate-only monitoring reference included in the bundle.

The manifest binds the model, report, policy, predictions, HTML report, and
monitoring reference by SHA-256. Report and manifest also agree on the optional
data-quality sidecar digest. Loading fails when a file is missing, changed,
incompatible with the runtime, or inconsistent with the feature contract. The
manifest is not cryptographically signed.

## Interpretation limits

- The records are historical and reflect social and economic inequality.
- Complete-case filtering can alter who remains in the dataset.
- Repeated feature vectors and cross-split overlap complicate independence and
  can support conflicting labels, but do not identify duplicate people.
- Income is a poor proxy for qualification, ability, or job performance.
- Protected fields are coarse, binary in places, and incomplete.
- Small intersectional positive or negative cells make some rates unstable or
  not estimable.
- Weighted sensitivity does not repair target mismatch, sampling limitations,
  or missing constructs.
- Reusing one test partition across seeds does not create independent samples
  from a current population.
