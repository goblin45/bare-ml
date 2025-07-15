from Neuron import Neuron
from Optimizer import Optimizer
from typing import List

class Layer:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, x: List[float]) -> List[float]:
        return [neuron.forward(x) for neuron in self.neurons]

    def compute_gradients(self, xs: List[float], errors: List[float]) -> None:
        for i, neuron in enumerate(self.neurons):
            neuron.compute_gradients(xs, errors[i])

    def apply_gradients(self, optimizer: Optimizer, learning_rate: float) -> None:
        for neuron in self.neurons:
            optimizer.step(neuron.weights, neuron.gradients, learning_rate)