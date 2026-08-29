# Responsible AI protocol

## Purpose and boundary

This repository studies a fairness intervention on UCI Adult, a 1994
census-income dataset. It is a benchmark evaluation artifact, not a model of
candidate quality and not a hiring system.

## Question under study

The experiment asks a narrow question: on one fixed benchmark, what happens to
measured sex-group disparities when the decision threshold for the lower-TPR
group is reduced using validation labels only?

The implementation is best described as **one-sided opportunity uplift**. It
keeps the Male threshold at `0.500` and may lower the Female threshold when the
validation TPR is lower. If the Female validation TPR is already at least the
Male TPR, it applies no adjustment. This is narrower than a general claim of
fairness or full bidirectional equalized-odds optimization.

## Protocol

1. Rebuild the cleaned data from the bundled UCI train and test files.
2. Preserve the official test partition.
3. Draw validation rows only from the original training partition, jointly
   stratified by income, sex, and binary race grouping.
4. Fit preprocessing and the classifier on fit rows only.
5. Tune the one-sided sex threshold on validation labels only.
6. Freeze thresholds and evaluate baseline and adjusted predictions once on
   the official test partition.
7. Persist the full report, policy, manifest, model, predictions, hashes,
   subgroup cells, and governance verdict as one versioned bundle.

Protected attributes are retained for offline evaluation but excluded from the
12 model features. The local API serves only the baseline `0.500` global
threshold and does not accept `sex` or `race` as inputs.

## Measurements

The report records:

- accuracy, precision, recall, F1, ROC AUC, PR AUC, and Brier score;
- statistical parity difference (SPD);
- disparate impact (DI);
- privileged and unprivileged true-positive rates and their gap;
- per-group sample, positive-label, negative-label, predicted-positive, TPR,
  and FPR diagnostics for sex, binary race, and their intersection;
- paired, label-and-group-stratified bootstrap intervals for baseline,
  adjusted, and change metrics.

An undefined rate is represented explicitly rather than silently replaced with
zero. Small cells remain visible in the report and must be interpreted with
caution.

## Reference outcome

For XGBoost, seed 42, and 500 paired bootstrap replicates:

| Metric | Baseline | Offline adjusted | Change |
|---|---:|---:|---:|
| Accuracy | 0.8694 | 0.8678 | -0.0016 |
| TPR gap | 0.0504 | -0.0124 | -0.0628 |
| SPD | 0.1754 | 0.1563 | -0.0191 |
| DI | 0.3400 | 0.4120 | +0.0720 |

The adjusted 95% bootstrap interval is `[-0.0560, 0.0268]` for TPR gap,
`[0.1468, 0.1670]` for SPD, `[0.3817, 0.4370]` for DI, and
`[0.8630, 0.8728]` for accuracy.

These results show a trade-off, not a success certificate. The default policy
gate **fails** because DI remains below `0.80` and absolute SPD remains above
`0.10`.

## Experimental policy gate

| Check | Default |
|---|---:|
| Adjusted accuracy | at least 0.80 |
| Accuracy drop from baseline | at most 0.02 |
| Absolute TPR gap | at most 0.05 |
| Disparate impact | 0.80 to 1.25 |
| Absolute SPD | at most 0.10 |

The strict parser rejects missing, malformed, non-finite, Boolean, and
out-of-domain metrics. CI rebuilds data, generates a real Logistic Regression
report through the canonical pipeline, and confirms the expected gate
rejection. Unit fixtures separately exercise pass and fail cases.

A pass would mean only that these chosen numeric checks passed for one dataset,
split, and model. It would not establish construct validity, job relatedness,
external validity, legality, individual fairness, privacy, security, or safe
operation.

## Known limitations

- Adult predicts an income label, not qualification, performance, or hiring
  outcomes.
- Its 1994 data reflects historical conditions and structural inequities.
- Binary sex and race encodings erase identities and within-group variation.
- The threshold policy uses a protected attribute offline and could be unlawful
  or inappropriate in a real decision process.
- Post-processing cannot repair biased labels, missing constructs, proxy
  discrimination, or data-collection harms.
- One seed and one model do not establish intervention stability.
- Bootstrap intervals capture test-sample uncertainty, not dataset shift or
  external validity.
- The repository has no stakeholder process, legal review, appeal mechanism,
  monitoring operation, or accountable decision owner.

## Misuse prevention

Do not describe the output as candidate quality, use it to rank applicants, or
deploy it for employment decisions. Any real project would require a valid and
job-related target, representative prospective data, independent legal and
domain review, affected-stakeholder participation, privacy and security
engineering, human accountability, contestability, and validation in the
intended context.

## Version

- Project version: `0.2.0`
- Reference run: `xgb-seed-42-v1`
- Last updated: 2026-08-28
