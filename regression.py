import time
from typing import List, Tuple

from LinearRegressor import LinearRegressor
from dataset import normalize_x

def linear_regression(
    train_data: Tuple[List[List[float]], List[float]],
    test_data: Tuple[List[List[float]], List[float]],
    learning_rate: float = 0.05,
    epochs: int = 10
) -> None:
    if len(train_data) < 2:
        raise ValueError("train_data must be a matrix of x and a list of corresponding y in train_data")
    if len(test_data) < 2:
        raise ValueError("test_data must be a matrix of x and a list of corresponding y in test_data")
    if epochs < 1:
        raise ValueError("Number of epochs must be a positive integer")
    if learning_rate < 0:
        raise ValueError("Learning rate must be a positive float")

    # training part
    x_train, y_train = train_data
    x_train = normalize_x(x_train)[0] # normalize x values from 0 to 1
    linear_regressor = LinearRegressor(x_train, y_train, learning_rate)
    start_time = time.time()
    print("Created Single Layer Perceptron with training data.")
    linear_regressor.train(epochs)
    print(f"Applied Stochastic Gradient Descent. Trained on {len(x_train)} samples.")

    # testing part
    x_test, y_test = test_data
    x_test = normalize_x(x_test)[0] # normalize x values from 0 to 1
    y_pred, accuracy = linear_regressor.predict(x_test, y_test)
    end_time = time.time()
    print(f"Tested on {len(x_test)} samples.")
    # print(f"Real results: {y_test}")
    # print(f"Predicted results: {[round(y,3) for y in y_pred]}")
    print(f"Accuracy: {accuracy:.3f}%\n")

    # measure time elapsed
    print(f"Time elapsed: {(end_time - start_time):.3f} seconds.")