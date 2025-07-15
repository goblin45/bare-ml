from Vector import Vector
from typing import List

class Neuron:
    def __init__(self, num_inputs: int) -> None:
        self.weights = Vector()
        self.weights.generate(num_inputs + 1, (1, 10), True) # bias (w0) adds that extra 1
        self.gradients = Vector([0 for i in range(num_inputs + 1)])

    def forward(self, xs: List[float]) -> 'float':
        return self.weights.dot(Vector([1.0] + xs))

    def compute_gradients(self, xs: List[float], error: float) -> None:
        xs_bias = [1.0] + xs
        for i in range(len(self.weights)):
            self.gradients.replace(i, error * xs_bias[i])