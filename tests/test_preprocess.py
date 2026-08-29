"""Tests for preprocessing pipeline."""

import numpy as np


class TestBuildPreprocessingPipeline:
    """Tests for the preprocessing pipeline builder."""

    def test_returns_transformer(self, sample_adult_data):
        """Test that function returns a ColumnTransformer."""
        from sklearn.compose import ColumnTransformer

        from fairness_project.features.preprocessing import build_preprocessing_pipeline

        preprocess, num_cols, cat_cols = build_preprocessing_pipeline(sample_adult_data)

        assert isinstance(preprocess, ColumnTransformer)
        assert len(num_cols) > 0
        assert len(cat_cols) > 0

    def test_excludes_sensitive_columns(self, sample_adult_data):
        """Test that sensitive columns are excluded from features."""
        from fairness_project.features.preprocessing import build_preprocessing_pipeline

        preprocess, num_cols, cat_cols = build_preprocessing_pipeline(sample_adult_data)

        all_feature_cols = num_cols + cat_cols

        assert "sex" not in all_feature_cols
        assert "race" not in all_feature_cols
        assert "race_binary" not in all_feature_cols
        assert "income" not in all_feature_cols
        assert "split" not in all_feature_cols

    def test_includes_numeric_features(self, sample_adult_data):
        """Test that numeric features are included."""
        from fairness_project.features.preprocessing import build_preprocessing_pipeline

        preprocess, num_cols, cat_cols = build_preprocessing_pipeline(sample_adult_data)

        assert "age" in num_cols
        assert "education_num" in num_cols
        assert "hours_per_week" in num_cols

    def test_includes_categorical_features(self, sample_adult_data):
        """Test that categorical features are included."""
        from fairness_project.features.preprocessing import build_preprocessing_pipeline

        preprocess, num_cols, cat_cols = build_preprocessing_pipeline(sample_adult_data)

        assert "workclass" in cat_cols
        assert "education" in cat_cols
        assert "occupation" in cat_cols

    def test_pipeline_can_fit_transform(self, sample_adult_data):
        """Test that the pipeline can fit and transform data."""
        from fairness_project.features.preprocessing import build_preprocessing_pipeline

        df_train = sample_adult_data[sample_adult_data["split"] == "train"]
        X_train = df_train.drop(columns=["income", "sex", "race", "race_binary", "split"])

        preprocess, _, _ = build_preprocessing_pipeline(df_train)

        # Should not raise
        X_transformed = preprocess.fit_transform(X_train)

        assert X_transformed.shape[0] == len(X_train)
        assert X_transformed.shape[1] > 0

    def test_handles_unknown_categories(self, sample_adult_data):
        """Test that unknown categories are handled gracefully."""
        from fairness_project.features.preprocessing import build_preprocessing_pipeline

        df_train = sample_adult_data[sample_adult_data["split"] == "train"].copy()
        df_test = sample_adult_data[sample_adult_data["split"] == "test"].copy()

        # Add unknown category to test
        df_test.loc[df_test.index[0], "workclass"] = "Unknown_Category"

        X_train = df_train.drop(columns=["income", "sex", "race", "race_binary", "split"])
        X_test = df_test.drop(columns=["income", "sex", "race", "race_binary", "split"])

        preprocess, _, _ = build_preprocessing_pipeline(df_train)
        preprocess.fit(X_train)

        # Should not raise with handle_unknown="ignore"
        X_transformed = preprocess.transform(X_test)

        assert X_transformed.shape[0] == len(X_test)


class TestPreprocessingConsistency:
    """Tests for preprocessing consistency."""

    def test_same_output_dimensions(self, sample_adult_data):
        """Test that train and test have same number of features after transform."""
        from fairness_project.features.preprocessing import build_preprocessing_pipeline

        df_train = sample_adult_data[sample_adult_data["split"] == "train"]
        df_test = sample_adult_data[sample_adult_data["split"] == "test"]

        X_train = df_train.drop(columns=["income", "sex", "race", "race_binary", "split"])
        X_test = df_test.drop(columns=["income", "sex", "race", "race_binary", "split"])

        preprocess, _, _ = build_preprocessing_pipeline(df_train)
        preprocess.fit(X_train)

        X_train_transformed = preprocess.transform(X_train)
        X_test_transformed = preprocess.transform(X_test)

        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]

    def test_numeric_features_scaled(self, sample_adult_data):
        """Test that numeric features are scaled (mean ~0, std ~1)."""
        from fairness_project.features.preprocessing import build_preprocessing_pipeline

        df_train = sample_adult_data[sample_adult_data["split"] == "train"]
        X_train = df_train.drop(columns=["income", "sex", "race", "race_binary", "split"])

        preprocess, num_cols, _ = build_preprocessing_pipeline(df_train)
        X_transformed = preprocess.fit_transform(X_train)

        # First n columns should be numeric (StandardScaler)
        n_numeric = len(num_cols)
        if hasattr(X_transformed, "toarray"):
            numeric_transformed = X_transformed[:, :n_numeric].toarray()
        else:
            numeric_transformed = X_transformed[:, :n_numeric]

        # Check approximately standardized
        means = np.mean(numeric_transformed, axis=0)
        stds = np.std(numeric_transformed, axis=0)

        assert all(abs(m) < 0.5 for m in means), f"Means not centered: {means}"
        assert all(0.5 < s < 2.0 for s in stds), f"Stds not normalized: {stds}"
