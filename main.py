from regression import linear_regression as lr
from dataset import *

def main():
    # linear regression
    data : Tuple[List[List[float]], List[float]] = generate_dataset_for_linear_regression(1000)
    x_train, x_test, y_train, y_test = train_test_split(data)

    # call linear regression
    lr((x_train, y_train), (x_test, y_test))

if __name__ == '__main__':
    main()