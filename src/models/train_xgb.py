from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def train_xgb(preprocess, X_train, y_train):

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model = Pipeline([("preprocess", preprocess), ("xgb", xgb)])

    model.fit(X_train, y_train)
    return model
