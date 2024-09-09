from enum import Enum
import numpy as np
from dataclasses import dataclass
import typing as tp
from neat.activations import Activations
from neat.init_schemes import InitSchemes
from neat.genes import NodeGene, NodeType, ConnectionGene


class Phylogeny:
    def __init__(
        self,
        population: int,
        inputs: int,
        outputs: int,
        weight_mutation_rate: float = 0.8,
        weight_perturb_rate: float = 0.9,
        link_mutate_rate_small: float = 0.05,
        link_mutate_rate_big: float = 0.3,
        node_mutate_rate: float = 0.03,
        disable_on_inherit_rate: float = 0.75,
        node_mutation_rate: float = 0.03,
        inter_species_mating_rate: float = 0.001,
        crossover_rate: float = 0.75,
        compatibility_scale_1: int = 1,
        compatibility_scale_2: int = 1,
        compatibility_scale_3: int = 0.4,
        compatibility_threshold: float = 3.0,
        activation: tp.Callable[[float], float] = Activations.neaty_sigmoid,
    ):
        self.genome1 = genome1
        self.genome2 = genome2

    def crossover(self) -> Genome:
        inputs = self.genome1.inputs
        outputs = self.genome1.outputs
        hidden = self.genome1.hidden

        connections = []
        for conn1, conn2 in zip(self.genome1.connections, self.genome2.connections):
            if conn1.enabled and conn2.enabled:
                if np.random.uniform() < 0.5:
                    connections.append(conn1)
                else:
                    connections.append(conn2)
            elif conn1.enabled:
                connections.append(conn1)
            elif conn2.enabled:
                connections.append(conn2)

        return Genome(
            inputs=inputs,
            outputs=outputs,
            hidden=hidden,
            connections=connections,
            activation=self.genome1.activation,
        )
