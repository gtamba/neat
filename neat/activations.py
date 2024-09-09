import numpy as np


class Activations:
    @staticmethod
    def neaty_sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-4.9 * x))

    @staticmethod
    def relu(x: float) -> float:
        return max(0, x)

    @staticmethod
    def tanh(x: float) -> float:
        return np.tanh(x)
