# Architecture

## Design objective

The system is organized around one invariant: a result is useful only when the
data contract, policy choice, held-out evidence, and artifact provenance remain
connected.

The canonical paths are `fairness audit` for one run and `fairness study` for
repeated-seed sensitivity. There is no alternate notebook or legacy training
path.

## Components

```text
                          +----------------------+
                          | configs/default.yaml |
                          +----------+-----------+
                                     |
                                     v
+-----------+    +------------+    +---------------------+
| UCI files | -> | preprocess | -> | strict data schema  |
+-----------+    +--+------+--+    +----------+----------+
                    |      |
                    |      `--> model-ready CSV
                    `---------> quality sidecar + source/data digests
                                                |
                                                v
                                      +-------------------+
                                      | split coordinator |
                                      +----+---------+----+
                                           |         |
                                      fit + val      | official test
                                           |         |
                                           v         |
                                  +----------------+ |
                                  | fitted pipeline| |
                                  +-------+--------+ |
                                          |          |
                          validation proba |          | held-out proba
                                          v          v
                 +--------------------+   +--------------------+
                 | threshold frontier |   | paired evaluation  |
                 +--------------------+   +--------------------+
                 +--------------------+             |
                 | global review band |             |
                 +--------------------+             |
                          |                         |
                          +------------+------------+
                                       |
                                       v
               +--------------------------------------------------+
               | uncertainty + intersections + weight sensitivity |
               | exact-overlap removal + novel-only fixed metrics |
               +-------------------------+------------------------+
                                       |
                                       v
                              +----------------+
                              | evidence gate  |
                              +-------+--------+
                                      |
                                      v
                           +-----------------------+
                           | atomic run publisher  |
                           +-----+------------+----+
                                 |            |
                                 v            v
                         seven bound files   validated loader
                              |               |          |
                              v               v          v
                     monitoring reference  v2 simulator offline compare
```

## Contract boundaries

### 1. Configuration

Configuration schema `2.0` rejects unknown top-level and nested keys. Semantic
validation runs before training. Each report and manifest contains the fully
resolved configuration and its SHA-256, not only the path to a YAML file.

Each run starts from a deep configuration clone. The effective model type and
validation ratio are written back to `model.model_type` and `data.val_size` in
that clone. The run seed is written to both `seed` and `model.random_state` and
is also passed to the splitter and estimator. The source `Config` object remains
unchanged. The synchronized clone is the configuration that is serialized and
hashed.

### 2. Data semantics

Preprocessing audits the raw files before complete-case deletion and the
model-ready table after cleaning. The `.quality.json` sidecar binds both source
files and the CSV by SHA-256. It records attrition, missingness, group
composition, exact and feature-vector duplicates, conflicting labels,
cross-split overlap, small-group flags, and `fnlwgt` semantics.

Experiment startup recomputes the processed audit. A present sidecar must match
the CSV digest and the recomputed evidence. Its raw and processed sections are
embedded in the report, and its digest is bound across report and manifest.

### 3. Model-ready data

The data schema requires the complete model-ready table with exact columns,
integer and string dtype families, non-null values, categorical domains, and
numeric ranges. Unexpected columns fail validation.

The official UCI test marker is preserved. A deterministic validation draw is
made only from the original training rows.

### 4. Scoring features

`FEATURE_COLUMNS` is an ordered 11-field contract. `fnlwgt` and protected audit
fields are outside it. The fitted model, manifest, simulation request, and
compatibility probe must all agree on that order.

The canonical numeric transformer receives, in order, `age`, `education_num`,
`capital_gain`, `capital_loss`, and `hours_per_week`. The canonical categorical
transformer receives, in order, `workclass`, `education`, `marital_status`,
`occupation`, `relationship`, and `native_country`. The loader checks both
transformer assignments, `handle_unknown="ignore"`, one nonempty unique fitted
vocabulary per categorical field, and exact agreement between fitted and
manifest-recorded transformed feature names.

The fitted vocabularies live inside the digest-bound `model.joblib`; their
derived transformed names are duplicated in report and manifest preprocessing
evidence. This ties evaluation transformation and the simulator allowlist to the
same training-fitted encoder rather than to a separate hand-maintained category
list.

Validation and test transformation intentionally ignore categories outside the
training vocabulary. The report and manifest retain split-level OOV evidence:
rows with any OOV value and, for each categorical field, training-vocabulary
size, distinct unknown values, affected rows, and shares. The simulator derives
its accepted categories from the same fitted encoder but rejects OOV input
before scoring.

### 5. Policy selection

Both policy choices use validation labels only:

- the offline frontier enumerates pairs of group thresholds;
- the simulation review band selects one global interval around the base
  threshold.

Neither function receives held-out labels during selection. The test partition
is used only after the choices are frozen.

The validation-dependence audit compares validation rows with fit rows using
exact 11-feature identities and also counts exact full-record overlap. Both
policy selectors are rerun on the overlap-excluded validation subset while the
model and validation probabilities stay fixed. These alternate policies remain
sensitivity evidence. They do not replace the canonical policies evaluated on
test. When exclusion leaves no rows or a selector cannot be estimated, the
reason is recorded instead of inventing a result.

### 6. Evaluation

Baseline and adjusted predictions are aligned on the same test rows. The report
contains performance, group fairness metrics, subgroup cells, intersectional
uncertainty, weighted sensitivity, review burden, exact-feature overlap
sensitivity, and optional paired bootstrap intervals. The overlap view compares
the complete held-out set with the novel-only subset while keeping every score,
prediction, and all policy thresholds frozen. It never retunes on held-out rows.
It also embeds the data-semantics evidence and a summary of the aggregate
monitoring reference.

### 7. Governance

The gate first validates report structure. Only a structurally valid report can
be evaluated against thresholds. Point metrics, interval bounds,
intersectional spans, accuracy loss, and held-out review constraints can all
contribute violations.

The exact threshold set used for a run is persisted inside the governance
verdict. A later gate call without an override reloads that policy rather than
silently falling back to defaults. Bundle save and load both run a fresh gate
evaluation and require the recomputed verdict, stored report verdict, and
manifest verdict to agree.

### 8. Artifact publication

The model, policy, report, predictions, HTML, and aggregate monitoring snapshot
are written to a unique sibling temporary directory with a fresh UUID suffix.
Their hashes are inserted into the manifest. Document bindings, fresh gate
evaluation, policy-to-report evidence, and the monitoring schema are validated
before one atomic rename publishes the seven-file run directory. Failures
remove the temporary directory and never expose a partial bundle.

The same collision-free sibling-path rule applies to every other writer that is
described as atomic: preprocessing CSV and JSON, general JSON and monitoring
outputs, batch CSV, and the incomplete-study marker. A direct HTML export is a
separate non-atomic convenience path.

### 9. Artifact loading

Loading is more than deserialization. It verifies:

- required files and digests;
- schema and cross-document identity;
- Git, source, data, data-quality, configuration, runtime, and dependency
  metadata;
- feature columns and contract ID;
- canonical numeric and categorical transformers, fitted training vocabulary,
  transformed names, and model parameters;
- model classes and positive-class mapping;
- fitted feature names;
- the monitoring snapshot's exact aggregate-only schema and digest;
- serving and offline-policy separation;
- serving policy agreement with selective-review evidence;
- offline policy agreement with validation selection, thresholds, and protocol;
  and
- a fresh gate verdict under the persisted threshold set.

After loading, `InferenceService` runs a live canonical-row compatibility probe
and requires one finite two-class probability result before accepting requests.

### 10. Simulation

The HTTP and CSV paths share `InferenceService`. Input validation rejects any
field outside the scoring plane and checks fitted categorical vocabularies. The
service rejects unseen categories before calling `predict_proba` and returns a
global review-band decision with artifact and policy identity.

Governance rejection is a loader error unless the caller opts into a local
research simulation.

### 11. Offline monitoring

The canonical run writes an aggregate reference from held-out features, baseline
scores and predictions, protected-group composition, delayed labels, and
`fnlwgt` sensitivity. `fairness monitor snapshot` applies the same strict role
schema to another offline CSV. `fairness monitor compare` validates
comparability, calculates descriptive feature, outcome, group, and optional
label drift, then returns `PASS`, `FAIL`, or `INSUFFICIENT_EVIDENCE`.

Snapshot validation is relational as well as structural. It checks category
shares and unknown summaries against counts, prediction rates against class
counts, delayed-label class counts against confusion totals, protected-group
totals against overall totals, and binary rates against their confusion
denominators. Contradictory aggregates are rejected before publication, load,
or comparison.

No row-level data is serialized in a snapshot. The comparison is separate from
the model-policy gate and does not alter or approve the run artifact.

## Repeated-seed orchestration

`fairness study` creates one complete run bundle per seed. Aggregation refuses
reports that differ in source, data, model type or parameters, resolved
configuration after removing only per-run seed fields, protocol, feature
contract, gate threshold policy, or metric coverage. If generation fails, the
study directory receives an `INCOMPLETE` marker rather than a partial summary.

The completed study writes:

```text
studies/<study-id>/
|-- runs/             complete constituent bundles
|-- stability.json    comparable metric and threshold distributions
`-- audit.html        worst-seed report with stability context
```

## Trust model

SHA-256 binding detects missing files, corruption, and partial or accidental
mutation. The data-quality sidecar connects raw sources to the model-ready CSV,
and the run manifest connects the monitoring reference to the model evidence.
Runtime checks prevent loading a serialized model in a mismatched environment.
Atomic publication prevents readers from seeing half a bundle.

The system does not use digital signatures, a transparency log, remote
attestation, or access-controlled artifact storage. A party able to replace the
entire bundle can replace its hashes and metadata too.

## Extension points

Safe extensions should preserve the same boundaries:

- add an estimator behind the model factory and record every parameter;
- add a metric as a report field before making it gate-relevant;
- add a group dimension without hiding low-support cells;
- change identity sensitivity only if exact canonical equality and frozen-policy
  evaluation remain explicit;
- add a policy selector only if selection and held-out evaluation remain
  separate;
- add a runtime interface only through the validated bundle loader; and
- extend monitoring only through the strict aggregate schema, explicit evidence
  status, and manifest-bound reference artifact.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the required tests.
