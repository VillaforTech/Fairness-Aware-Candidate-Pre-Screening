# Data Documentation

## Adult (Census Income) Dataset

### Source

The Adult dataset is from the UCI Machine Learning Repository:
- **URL**: https://archive.ics.uci.edu/ml/datasets/adult
- **Original source**: 1994 Census database
- **Donated by**: Ronny Kohavi and Barry Becker (Silicon Graphics)

### License

The UCI Machine Learning Repository datasets are available under:
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- You are free to share and adapt the data for any purpose
- Attribution required: Cite the UCI repository and original paper

### Citation

If you use this dataset, please cite:

```bibtex
@misc{kohavi1996adult,
  author = {Kohavi, Ronny},
  title = {Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid},
  year = {1996},
  publisher = {AAAI Press},
  booktitle = {Proceedings of the Second International Conference on Knowledge Discovery and Data Mining}
}
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

For fairness analysis, we consider:

1. **sex**: Binary (Male/Female)
   - Privileged group: Male
   - Unprivileged group: Female

2. **race_binary**: Binary (White/Non-White)
   - Derived from original 'race' column
   - Privileged group: White
   - Unprivileged group: Non-White

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

```
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

### Ethical Considerations

This dataset is used for **educational and research purposes** to study algorithmic fairness. When applying similar techniques to real hiring decisions:

1. **Legal compliance**: Ensure compliance with employment discrimination laws
2. **Stakeholder involvement**: Include affected communities in model development
3. **Regular auditing**: Continuously monitor for fairness issues
4. **Human oversight**: ML predictions should inform, not replace, human decisions
5. **Documentation**: Maintain clear records of model decisions and their basis
