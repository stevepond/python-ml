import numpy as np
from ml_app.data_loader import load_iris_data
from ml_app.model import train_model, evaluate_model, tune_model


def test_train_and_evaluate():
    X_train, X_test, y_train, y_test = load_iris_data()
    clf = train_model(X_train, y_train)
    acc = evaluate_model(clf, X_test, y_test)
    assert acc > 0.7


def test_tune_model():
    X_train, _, y_train, _ = load_iris_data()
    param_grid = {"C": [0.1, 1.0], "solver": ["lbfgs"]}
    search = tune_model(X_train, y_train, param_grid)
    assert set(search.best_params_.keys()) == set(param_grid.keys())
