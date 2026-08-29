# Model Card: UCI Adult Fairness Audit

> Generated from validated run `xgb-seed-42-v1`.
> This is a benchmark evaluation artifact, not a hiring model.

## Provenance

- Model: `xgb`
- Created: 2026-08-28T22:40:08.399728+00:00
- Git commit: `548223e92a62853c2a991f093a83a34c4fa44b6d`
- Data SHA-256: `2496fc2982288003e36bdf8f2c41324e661ef959afe4151b2bfc70f2dc20e9f9`
- Source SHA-256: `e65ff2d0205531f8c275bfdd9d9a99ce7eb3e2d3c181a711b1692fd0339dd62c`
- Seed: 42
- Dirty worktree recorded: True

## Evaluation protocol

- Dataset: UCI Adult (1994 Census income classification)
- Validation: joint stratification by income, sex, and race_binary
- Rows: 25,637 fit / 4,525 validation / 15,060 test
- EO thresholds were tuned on validation labels only.
- Final metrics were computed once on the preserved official test partition.
- Protected attributes were excluded from model features.
- Frozen offline thresholds: Male `0.500`, Female `0.405`.

## Results

| Metric | Baseline | Offline adjusted | Change |
|---|---:|---:|---:|
| accuracy | 0.8694 | 0.8678 | -0.0016 |
| precision | 0.7759 | 0.7641 | -0.0117 |
| recall | 0.6586 | 0.6681 | +0.0095 |
| f1 | 0.7125 | 0.7129 | +0.0004 |
| SPD | 0.1754 | 0.1563 | -0.0191 |
| DI | 0.3400 | 0.4120 | +0.0720 |
| TPR_gap | 0.0504 | -0.0124 | -0.0628 |

### Adjusted 95% paired-bootstrap intervals

| Metric | Lower | Upper |
|---|---:|---:|
| accuracy | 0.8630 | 0.8728 |
| SPD | 0.1468 | 0.1670 |
| DI | 0.3817 | 0.4370 |
| TPR_gap | -0.0560 | 0.0268 |

## Experimental policy gate

**FAILED**

- DI=0.4120 < min_disparate_impact=0.8
- |SPD|=0.1563 > max_spd=0.1

## Serving boundary

The local API serves the baseline global threshold only. It does not apply the
offline sex-specific thresholds. API responses name the policy and artifact ID
so the offline fairness experiment cannot be mistaken for deployed behavior.

## Limitations

- Adult is a 1994 census-income dataset, not applicant or job-performance data.
- Binary sex and race groupings erase identity and intersectional detail.
- Bootstrap intervals describe test-sample uncertainty, not external validity.
- Passing a configurable gate would not establish safety, legality, or validity.

## Authors

Roberto Villafuerte and Charles Santhakumar, University of Helsinki collaboration.
