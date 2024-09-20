from neat.genes import Genome
from neat.phylogeny import Phylogeny
import numpy as np
import logging


def xor_fitness(g: Genome) -> float:
    inputs = [
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]

    outputs = [0, 1, 1, 0]

    fitness = 0
    for i, o in zip(inputs, outputs):
        pred = g.forward(i)[0]
        # pred = 1 if pred > 0.5 else 0
        fitness += abs(pred - o) ** 2

    return 4 - fitness


if __name__ == "__main__":
    import numpy as np

    np.random.seed(3)
    logging.basicConfig(level=logging.INFO)
    xor_phylogeny = Phylogeny(
        population_size=150,
        input_size=2,
        output_size=1,
        fitness_function=xor_fitness,
    )

    xor_phylogeny.evolve(log=True)
    print("done")
