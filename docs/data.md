# Data Documentation

## Adult (Census Income) Dataset

### Source

The Adult dataset is from the UCI Machine Learning Repository:

- **URL**: <https://archive.ics.uci.edu/dataset/2/adult>
- **Original source**: 1994 Census database
- **Donated by**: Ronny Kohavi and Barry Becker (Silicon Graphics)

### License

The Adult dataset page lists:

- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- You are free to share and adapt the data for any purpose
- Attribution required: Cite the UCI repository and original paper

### Citation

If you use this dataset, please cite:

```text
Becker, B. & Kohavi, R. (1996). Adult [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
```

### Dataset Description

**Task**: Binary classification - predict whether income exceeds $50K/year

**Samples**: 48,842 total (32,561 training + 16,281 test)

**Features**: 14 attributes (6 continuous, 8 categorical)

### Columns

| Column | Type | Description |
|--------|------|-------------|
| age | continuous | Age of individual |
| workclass | categorical | Type of employment |
| fnlwgt | continuous | Final sampling weight |
| education | categorical | Highest education level |
| education_num | continuous | Education level as number |
| marital_status | categorical | Marital status |
| occupation | categorical | Type of occupation |
| relationship | categorical | Relationship status |
| race | categorical | Race/ethnicity |
| sex | categorical | Biological sex (Male/Female) |
| capital_gain | continuous | Capital gains |
| capital_loss | continuous | Capital losses |
| hours_per_week | continuous | Hours worked per week |
| native_country | categorical | Country of origin |
| income | target | Income class (>50K, <=50K) |

### Sensitive Attributes

The processed data retains these groupings. The documented leakage-free command
computes and tunes its reported fairness metrics only for `sex`; a race analysis
would need to be run and reported separately.

1. **sex**: Binary (Male/Female)
   - Privileged group: Male
   - Unprivileged group: Female

2. **race_binary**: Binary (White/Non-White)
   - Derived from original 'race' column
   - Available for separate analysis; not evaluated by the documented smoke run

### Data Processing

1. **Download**: Raw data from UCI repository
2. **Clean**: Remove missing values, standardize labels
3. **Split**: Use original train/test split (preserved in 'split' column)
4. **Validation**: For EO threshold tuning, create val split from training data

### Known Issues

1. **Historical bias**: Data from 1994 reflects historical disparities
2. **Missing values**: ~7% of rows have missing values (handled by removal)
3. **Feature selection**: fnlwgt is sampling weight, not a predictive feature
4. **Label imbalance**: ~24% positive class (>50K)

### File Structure

```text
data/
├── raw/adult/
│   ├── adult.data      # Training data
│   ├── adult.test      # Test data
│   └── adult.names     # Column descriptions
└── processed/adult/
    └── adult_model_ready.csv  # Cleaned, combined data
```

### Usage

```python
# Download data
from fairness_project.data.download import download_adult_dataset
download_adult_dataset()

# Preprocess data
from fairness_project.data.preprocess import prepare_model_ready_data
df = prepare_model_ready_data()

# Validate schema
from fairness_project.data.schema import validate_dataframe
errors = validate_dataframe(df)
```

### API Input Schema

The prediction API accepts 12 fields. Protected attributes (`sex`, `race`) and the target (`income`) are intentionally excluded from the input schema.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `age` | int | 0-120 | Age of individual |
| `workclass` | string | required | Type of employment (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked) |
| `fnlwgt` | int | >= 0 | Final sampling weight |
| `education` | string | required | Highest education level (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool) |
| `education_num` | int | 1-20 | Education level as number |
| `marital_status` | string | required | Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse) |
| `occupation` | string | required | Type of occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces) |
| `relationship` | string | required | Relationship status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried) |
| `native_country` | string | required | Country of origin |
| `capital_gain` | int | >= 0 | Capital gains |
| `capital_loss` | int | >= 0 | Capital losses |
| `hours_per_week` | int | 0-168 | Hours worked per week |

**Unknown value handling**: The preprocessing pipeline uses `OneHotEncoder` with `handle_unknown="ignore"`, so unseen categorical values are mapped to a zero vector rather than causing errors.

### Data Lineage

**Training flow**:

```text
UCI Repository → Download (raw/) → Preprocess (processed/) → Train/Val/Test Split
→ Feature Engineering (StandardScaler + OneHotEncoder) → Model Training
→ Test Evaluation → Prediction and Metric CSV Files
```

**Intended API-scaffold flow**:

```text
User Input → Pydantic Validation → DataFrame → Model.predict() → Response → Audit Log
```

This flow is not currently runnable end to end with the tracked model artifact;
see [API Scaffold and Operations Notes](deployment.md).

The verified leakage-free script records CSV outputs but does not create a
versioned model registry or run manifest. Record the commit, Python environment,
seed, and command separately when reporting results.

### Ethical Considerations

This dataset is used for **educational and research purposes** to study
algorithmic fairness. These results do not justify use in real hiring decisions.
A separate real-world project would require, at minimum:

1. **Legal compliance**: Ensure compliance with employment discrimination laws
2. **Stakeholder involvement**: Include affected communities in model development
3. **Regular auditing**: Continuously monitor for fairness issues
4. **Human oversight**: ML predictions should inform, not replace, human decisions
5. **Documentation**: Maintain clear records of model decisions and their basis
