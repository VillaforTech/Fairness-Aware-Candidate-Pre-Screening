from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def train_rf(preprocess, X_train, y_train):
    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    rf_model.fit(X_train, y_train)
    return rf_model
