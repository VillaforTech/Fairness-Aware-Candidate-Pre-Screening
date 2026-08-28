# Responsible AI Protocol

## Purpose

This document describes fairness checks implemented in an educational,
hypothetical pre-screening scenario. It is not deployment guidance for an
employment system.

## Fairness Goals

### Primary Objective

Study measured group disparities in this dataset and scenario by:

1. Measuring disparities in model predictions
2. Applying post-processing techniques to reduce unfair gaps
3. Documenting trade-offs between accuracy and fairness

### Sensitive Attributes

- **Reported run**: Sex (Male/Female)
- **Available but not reported by the documented smoke run**: Race
  (White/Non-White binary grouping)

### Fairness Metrics Monitored

- **Statistical Parity Difference (SPD)**: Measures difference in positive prediction rates
- **Disparate Impact (DI)**: Ratio of positive rates between groups
- **Equal Opportunity (TPR Gap)**: Difference in true positive rates

## Mitigation Techniques

### Equal Opportunity Post-Processing

- Adjusts decision thresholds per group to equalize true positive rates
- Thresholds are tuned on validation data to prevent test set leakage
- Trade-off: May slightly reduce overall accuracy

### Implementation Protocol

1. **Train/Val/Test Split**: Separate validation set for threshold tuning
2. **Threshold Tuning**: Done ONLY on validation data
3. **Evaluation**: Final metrics computed on held-out test data
4. **Documentation**: The script saves prediction and metric CSV files and
   prints tuned thresholds to the terminal; it does not persist a complete run
   manifest

## Known Limitations

### Data Limitations

- Historical bias: Data reflects 1994 census patterns
- Binary groupings: Simplifies race to White/Non-White
- Missing intersectional analysis: Does not examine combined effects

### Model Limitations

- Post-processing only: Does not address in-processing or pre-processing fairness
- Binary classification: Limited to >50K income threshold
- Static thresholds: May not adapt to distribution shifts

### Evaluation Limitations

- Limited metrics: Does not include all possible fairness criteria
- Single sensitive attribute at a time: No intersectionality support
- Proxy discrimination: Does not detect indirect discrimination

## Ethical Considerations

### Potential Harms if Misused

- The Adult target is income, not job qualification, so interpreting its output
  as candidate quality would be invalid.
- If repurposed for decisions about people, errors and historical bias could
  deny opportunities or distribute them unevenly.
- Equalizing one group metric does not establish individual fairness or remove
  other forms of harm.

### Requirements Before Any Real-World Project

This artifact should not be deployed for employment decisions. A separate
real-world project would need, at minimum, a valid target and dataset,
independent legal and domain review, affected-stakeholder input, security and
privacy engineering, human accountability, contestability, and prospective
validation in the intended context.

## Compliance Considerations

The repository makes no compliance claim and does not implement a compliance
program. Applicable obligations vary by place, data, and use and can change;
obtain current qualified advice before considering any real-world system.

## Unimplemented Real-World Controls

The repository contains experimental drift utilities, but it does not provide a
validated monitoring service, alerting path, incident-response process, appeal
mechanism, or accountable operating team.

## Governance as Code

The experimental gate compares supplied metrics with configured thresholds. A
passing result does not establish that a model is suitable for deployment.

### Experimental Threshold Fixture

For demonstrations, the checker compares a supplied report with these defaults:

| Check | Threshold |
|-------|-----------|
| Minimum accuracy | >= 0.80 |
| Maximum TPR gap | <= 0.05 |
| Minimum disparate impact | >= 0.80 |
| Maximum SPD | <= 0.10 |

### Automated CI Checks

CI supplies fixed passing and failing fixtures to verify the checker. It does
not train a model or gate a deployable artifact.

### Configuration Changes

There is no implemented approval or override workflow. Threshold changes are
explicit experiment configuration and should be recorded with their rationale;
changing them does not validate a model for use.

For full governance documentation, see [Governance Gate Documentation](governance.md).

## Version Control

- **Current Version**: 0.1.0
- **Last Updated**: 2026-08-28

## Contact

For questions about this responsible AI protocol, contact the project maintainers.
