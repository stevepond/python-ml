from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a simple logistic regression model."""
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate the model on the test set and return accuracy."""
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Tuple],
    cv: int = 5,
) -> GridSearchCV:
    """Perform grid search to tune hyperparameters."""
    base_clf = LogisticRegression(max_iter=200)
    search = GridSearchCV(base_clf, param_grid, cv=cv)
    search.fit(X_train, y_train)
    return search
