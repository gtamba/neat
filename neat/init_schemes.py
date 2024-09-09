import numpy as np


class InitSchemes:
    # A useless initialization to make testing easy rotfl
    @staticmethod
    def one(input_nodes: int, output_nodes: int) -> int:
        return 1

    @staticmethod
    def uniform(input_nodes: int, output_nodes: int) -> int:
        return np.random.uniform(-1, 1)

    @staticmethod
    def uniform_scaled(input_nodes: int, output_nodes: int) -> int:
        return np.random.uniform(-1, 1) / max(input_nodes, output_nodes)
