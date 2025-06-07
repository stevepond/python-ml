"""Example application demonstrating training and tuning a model."""
from pprint import pprint

from ml_app.data_loader import load_iris_data
from ml_app.model import train_model, evaluate_model, tune_model


def main() -> None:
    X_train, X_test, y_train, y_test = load_iris_data()
    clf = train_model(X_train, y_train)
    acc = evaluate_model(clf, X_test, y_test)
    print(f"Initial model accuracy: {acc:.2f}")

    param_grid = {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]}
    search = tune_model(X_train, y_train, param_grid)
    print("Best parameters from grid search:")
    pprint(search.best_params_)
    tuned_acc = evaluate_model(search.best_estimator_, X_test, y_test)
    print(f"Tuned model accuracy: {tuned_acc:.2f}")


if __name__ == "__main__":
    main()
