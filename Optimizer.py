from Vector import Vector

class Optimizer:
    def step(self, weights: Vector, gradients: Vector, learning_rate: float):
        raise NotImplementedError()

class SGD(Optimizer):
    def step(self, weights: Vector, gradients: Vector, learning_rate: float):
        for i in range(len(weights)):
            weights.replace(i, weights.__getitem__(i) + learning_rate * gradients[i])