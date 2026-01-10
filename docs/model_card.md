# Model Card: Fairness-Aware Candidate Pre-Screening

## Model Details

### Overview
- **Model Name**: Fairness-Aware Income Classifier
- **Version**: 0.1.0
- **Model Type**: Binary Classification (XGBoost/Random Forest/Logistic Regression)
- **Framework**: scikit-learn, XGBoost

### Intended Use
- **Primary Use**: Educational demonstration of fairness-aware ML pipelines
- **Intended Users**: ML researchers, students, practitioners learning about algorithmic fairness
- **Out-of-Scope Uses**: Production hiring decisions without additional validation

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
- **Size**: ~48,000 samples (32,561 train + 15,055 test)
- **Collection Date**: 1994 U.S. Census
- **License**: CC BY 4.0

### Features
- **Numeric (6)**: age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week
- **Categorical (8)**: workclass, education, marital_status, occupation, relationship, race, sex, native_country
- **Target**: income (>50K vs <=50K)

### Preprocessing
- Missing values: Removed (~7% of data)
- Categorical encoding: One-hot encoding with unknown handling
- Numeric scaling: Standardization (mean=0, std=1)

## Evaluation

### Metrics

#### Performance (Test Set)
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 0.847 | 0.732 | 0.600 | 0.660 |
| Random Forest | 0.845 | 0.712 | 0.621 | 0.664 |
| XGBoost | 0.862 | 0.746 | 0.654 | 0.697 |

#### Fairness (Test Set, Before/After EO)
| Model | SPD Before | SPD After | TPR Gap Before | TPR Gap After |
|-------|------------|-----------|----------------|---------------|
| Logistic Regression | 0.175 | 0.154 | 0.070 | -0.002 |
| Random Forest | 0.181 | 0.162 | 0.066 | -0.002 |
| XGBoost | 0.170 | 0.155 | 0.065 | 0.000 |

### Evaluation Protocol
- **Split**: Train (80%) / Validation (15% of train) / Test (20%)
- **Leakage Prevention**: EO thresholds tuned on validation only
- **Reporting**: Final metrics on held-out test set

## Ethical Considerations

### Fairness
- **Protected Attributes**: sex, race
- **Mitigation Applied**: Equal Opportunity post-processing
- **Remaining Disparities**: SPD reduced but not eliminated

### Limitations
- Historical bias in training data (1994)
- Binary groupings may oversimplify
- No intersectionality analysis
- Post-processing only, not in-processing

### Potential Misuse
- Should not be used for actual hiring decisions without extensive validation
- Results may not generalize to modern populations
- Binary classification may not capture nuance in income prediction

## Usage

### Installation
```bash
pip install -e "."
```

### Quick Start
```python
# Load model
from fairness_project.models.registry import load_model
model = load_model("latest")

# Make predictions
predictions = model.predict(X_test)
```

### Citation
If using this model, please cite the UCI Adult dataset:
```
Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers.
```

## Maintenance

### Updates
- Model should be retrained if data distribution shifts significantly
- Fairness metrics should be monitored in production
- Thresholds may need recalibration over time

### Contact
For issues or questions, see the project repository.
