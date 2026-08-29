from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_lr(preprocess, X_train, y_train):
    lr_model = Pipeline(
        steps=[("preprocess", preprocess), ("model", LogisticRegression(max_iter=500))]
    )

    lr_model.fit(X_train, y_train)
    return lr_model
