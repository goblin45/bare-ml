import random
from typing import Tuple, List

def generate_dataset_for_linear_regression(
    data_points: int = 500,
    x_vector_length: int = 10
) -> Tuple[List[List[float]], List[float]]:
    bias = random.randint(0, 1000)
    weights = [random.randint(0, 1000) for _ in range(x_vector_length)]
    xs, ys = [], []
    for _ in range(data_points):
        xk = []
        yk = bias
        for i in range(x_vector_length):
            xk.append(random.randint(0, 1000))
            yk += weights[i] * xk[i]
        xs.append(xk)
        ys.append(yk)
    return xs, ys

def normalize_x(xs: List[List[float]]) -> Tuple[List[List[float]], List[Tuple[float, float]]]:
    num_features = len(xs[0])
    feature_mins = [min(col) for col in zip(*xs)]
    feature_maxs = [max(col) for col in zip(*xs)]
    normalized_xs = []
    for x in xs:
        norm_x = []
        for i in range(num_features):
            min_val = feature_mins[i]
            max_val = feature_maxs[i]
            if max_val == min_val:
                norm_x.append(0.0)
            else:
                norm_val = (x[i] - min_val) / (max_val - min_val)
                norm_x.append(norm_val)
        normalized_xs.append(norm_x)
    return normalized_xs, list(zip(feature_mins, feature_maxs))

def train_test_split(
    data: Tuple[List[List[float]], List[float]],
    split_ratio: float = 0.8
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    xs, ys = data
    split_idx = int(len(xs) * split_ratio)
    x_train, x_test = xs[:split_idx], xs[split_idx:]
    y_train, y_test = ys[:split_idx], ys[split_idx:]
    return x_train, x_test, y_train, y_test