# Responsible AI Protocol

## Purpose

This document outlines the responsible AI practices implemented in this fairness-aware
candidate pre-screening project. It serves as a guide for developers, stakeholders, and
auditors to understand how fairness considerations are integrated into the ML pipeline.

## Fairness Goals

### Primary Objective
Ensure equitable treatment across demographic groups in candidate pre-screening by:
1. Measuring disparities in model predictions
2. Applying post-processing techniques to reduce unfair gaps
3. Documenting trade-offs between accuracy and fairness

### Protected Attributes
- **Sex**: Male/Female
- **Race**: White/Non-White (binary grouping)

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
4. **Documentation**: All results and thresholds are logged

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

### Potential Harms
- **False negatives**: Qualified candidates may be incorrectly rejected
- **False positives**: Unqualified candidates may be incorrectly selected
- **Group-level vs. individual fairness**: Equalizing group rates may not ensure individual fairness

### Recommendations for Deployment
1. **Human oversight**: Use predictions as decision support, not final decisions
2. **Regular auditing**: Monitor fairness metrics on production data
3. **Feedback loops**: Collect outcomes to identify and correct errors
4. **Transparency**: Document model limitations for end users
5. **Appeal process**: Provide mechanism for candidates to contest decisions

## Compliance Considerations

### Regulatory Framework
This project is designed for educational purposes. Real-world deployment should consider:
- **Employment discrimination laws** (Title VII, EEOC guidelines)
- **Data protection regulations** (GDPR, CCPA)
- **AI-specific regulations** (EU AI Act proposals)

### Documentation Requirements
- Model cards should be maintained and updated
- Decision logs should be retained for audit purposes
- Impact assessments should be conducted before deployment

## Monitoring Plan

### Production Monitoring
- Track prediction distributions by sensitive group
- Monitor fairness metrics over time
- Alert on significant drift (PSI > 0.1)

### Incident Response
1. If fairness drift detected: Investigate root cause
2. If performance degradation: Retrain on recent data
3. If systematic errors: Update thresholds or retrain model

## Governance as Code

Automated governance gates enforce fairness and performance standards before any model can be deployed.

### Pre-Deployment Gate

Every model must pass these thresholds:

| Check | Threshold |
|-------|-----------|
| Minimum accuracy | >= 0.80 |
| Maximum TPR gap | <= 0.05 |
| Minimum disparate impact | >= 0.80 |
| Maximum SPD | <= 0.10 |

### Automated CI Checks

The governance gate runs as part of the CI pipeline. Both passing and failing scenarios are validated to ensure the gate works correctly.

### Override Policy

In exceptional cases, a governance gate failure can be overridden with:
1. Documented justification for why the threshold does not apply
2. Explicit sign-off from the responsible AI lead
3. A remediation plan with a timeline for meeting the threshold

For full governance documentation, see [Governance Gate Documentation](governance.md).

## Version Control

- **Current Version**: 0.1.0
- **Last Updated**: 2024
- **Review Schedule**: Quarterly or after significant changes

## Contact

For questions about this responsible AI protocol, contact the project maintainers.
