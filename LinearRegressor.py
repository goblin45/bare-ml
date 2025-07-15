from Layer import Layer
from Optimizer import SGD
from statistics import *
from array_utils import *
from typing import *

# let's try to make this layer solve linear regression problem
class LinearRegressor:
    def __init__(self, x_train: List[List[float]], y_train: List[float], learning_rate: float) -> None:
        # check if data is perfect to operate on
        check_training_data(x_train, y_train)

        self.layer = Layer(len(x_train[0]), 1)
        self.x_train: List[List[float]] = [x_train[i] for i in range(len(x_train))]
        self.y_train : List[float] = y_train
        self.learning_rate = learning_rate

    def predict_single(self, x_vector: List[float]) -> float:
        return self.layer.forward(x_vector)[0]

    def train(self, epochs):
        for _ in range(epochs):
            for x, y in zip(self.x_train, self.y_train):
                y_pred = self.predict_single(x)
                error = y - y_pred
                self.layer.compute_gradients(x, [error])
                self.layer.apply_gradients(SGD(), self.learning_rate)

    def predict(self, x_test: List[List[float]], y_test: List[float]) -> Tuple[List[float], float]:
        # check if test data is perfect to operate on
        check_testing_data(x_test, y_test, self.x_train)

        y_pred : List[float] = []
        error : float = 0.0
        for i in range(len(y_test)):
            pred = self.predict_single(x_test[i])
            if y_test[i] == 0:
                error = pred
            else:
                error += abs(pred - y_test[i]) / y_test[i]
            y_pred.append(pred)
        normalized_score : float = get_normalized_score_from_error(error, len(y_test))
        return y_pred, normalized_score

def check_training_data(x_train: List[List[float]], y_train: List[float]) -> None:
    if not row_has_data(x_train):
        raise ValueError('Empty x_train matrix is not allowed')
    all_rows_have_same_length(x_train)
    if not row_has_data(y_train):
        raise ValueError('Empty y_train matrix is not allowed')
    if len(x_train) != len(y_train):
        raise ValueError('x_train and y_train must have the same length')

def check_testing_data(x_test: List[List[float]], y_test: List[float], x_train: List[List[float]]) -> None:
    if not row_has_data(x_test):
        raise ValueError('Empty x_test matrix is not allowed')
    all_rows_have_same_length(x_test)
    if not row_has_data(y_test):
        raise ValueError('Empty y_test matrix is not allowed')
    if len(x_test) != len(y_test):
        raise ValueError('x_test and y_test must have the same length')
    if len(x_test[0]) != len(x_train[0]):
        raise ValueError('x_train and x_test must have the same length')