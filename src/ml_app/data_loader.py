from sklearn import datasets
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np

def load_iris_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the Iris dataset and split into train and test sets."""
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
