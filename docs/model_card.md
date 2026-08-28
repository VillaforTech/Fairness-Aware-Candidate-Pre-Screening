# Model Card: Fairness-Aware Candidate Pre-Screening

> The optional generator requires manually prepared `runs/<run_id>/run.json`
> and `runs/<run_id>/metrics.json` files:
>
> ```bash
> python -m fairness_project.evaluation.model_card --run-id <run_id>
> ```
>
> This will overwrite this file. The documented leakage-free pipeline does not
> create those run files automatically.

## Model Details

### Overview

- **Model Name**: Fairness-Aware Income Classifier
- **Version**: 0.1.0
- **Model Type**: Binary Classification (XGBoost/Random Forest/Logistic Regression)
- **Framework**: scikit-learn, XGBoost

### Intended Use

- **Primary Use**: Educational demonstration of fairness-aware ML pipelines
- **Intended Users**: ML researchers, students, practitioners learning about algorithmic fairness
- **Out-of-Scope Uses**: Employment screening or hiring decisions

### Model Architecture

- **Preprocessing**: StandardScaler (numeric) + OneHotEncoder (categorical)
- **Classifier Options**:
  - Logistic Regression (max_iter=500)
  - Random Forest (n_estimators=300)
  - XGBoost (n_estimators=300, max_depth=4)
- **Post-Processing**: Equal Opportunity threshold adjustment

## Training Data

### Dataset

- **Source**: UCI Adult (Census Income) Dataset
- **Size**: 48,842 raw rows; 45,222 after missing-value removal
  (30,162 original-train + 15,060 original-test rows)
- **Collection Date**: 1994 U.S. Census
- **License**: CC BY 4.0

### Features

- **Numeric (6)**: age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week
- **Categorical (8)**: workclass, education, marital_status, occupation, relationship, race, sex, native_country
- **Target**: income (>50K vs <=50K)

### Preprocessing

- Missing values: Rows removed (~7.4% of raw rows)
- Categorical encoding: One-hot encoding with unknown handling
- Numeric scaling: Standardization (mean=0, std=1)

## Evaluation

### Metrics

#### Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 0.8477 | 0.7322 | 0.5992 | 0.6590 |
| Random Forest | 0.8441 | 0.7103 | 0.6170 | 0.6604 |
| XGBoost | 0.8687 | 0.7723 | 0.6600 | 0.7117 |

#### Fairness (Test Set, Before/After EO)

| Model | SPD Before | SPD After | TPR Gap Before | TPR Gap After |
|-------|------------|-----------|----------------|---------------|
| Logistic Regression | 0.1764 | 0.1668 | 0.0713 | 0.0372 |
| Random Forest | 0.1790 | 0.1731 | 0.0691 | 0.0421 |
| XGBoost | 0.1760 | 0.1666 | 0.0584 | 0.0117 |

### Evaluation Protocol

- **Split**: 4,524 validation rows sampled from the 30,162 cleaned original
  training rows, leaving 25,638 fit rows; 15,060 cleaned original test rows
- **Leakage Prevention**: EO thresholds tuned on validation only
- **Reporting**: Final metrics on held-out test set

## Ethical Considerations

### Fairness

- **Reported Sensitive Attribute**: sex; race is retained in the data but was
  not evaluated in the documented run
- **Mitigation Applied**: Equal Opportunity post-processing
- **Remaining Disparities**: SPD reduced but not eliminated

### Limitations

- Historical bias in training data (1994)
- Binary groupings may oversimplify
- No intersectionality analysis
- Post-processing only, not in-processing

### Potential Misuse

- Must not be used to make employment or hiring decisions
- Results may not generalize to modern populations
- Binary classification may not capture nuance in income prediction

## Usage

### Installation

```bash
pip install -e "."
```

### Quick Start

```bash
fairness data preprocess
PYTHONPATH=src:. python src/main_leakage_free.py --seed 42
```

### Citation

When referring to the data, cite the UCI Adult dataset:

```text
Becker, B., & Kohavi, R. (1996). Adult [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
```

## Maintenance

### Updates

- Re-running the experiment on a different data context requires a fresh
  evaluation; this repository does not define a real-world retraining policy.

### Contact

For issues or questions, see the project repository.
