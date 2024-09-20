import bisect
from enum import Enum
import numpy as np
from dataclasses import dataclass
import typing as tp
from neat.activations import Activations
from neat.init_schemes import InitSchemes
from neat.genes import Genome, NodeGene, NodeType, ConnectionGene
import logging

logger = logging.getLogger(__name__)


@dataclass
class Species:
    id: int
    members: tp.List[Genome]
    adjusted_fitness: float = 0.0
    _representative: Genome = None

    def reset_members(self):
        self.members = []

    def reorganize(self):
        self.members.sort(key=lambda x: x.fitness, reverse=True)
        total = sum([g.fitness for g in self.members])
        # prob = (
        #     [g.fitness / total for g in self.members]
        #     if total > 0
        #     else [1 / len(self.members)] * len(self.members)
        # )
        self._representative = np.random.choice(self.members)
        self.adjusted_fitness = total / len(self.members)

    def compatibility(self, genome: Genome, c1: float, c2: float, c3: float) -> float:
        return Genome.distance(self.representative, genome, c1=c1, c2=c2, c3=c3)

    @property
    def champion(self):
        return max(self.members, key=lambda x: x.fitness)

    @property
    def representative(self):
        return self._representative

    @property
    def size(self):
        return len(self.members)

    def __repr__(self):
        return f"Species {self.id} \n Members: {len(self.members)} \n Adjusted Fitness: {self.adjusted_fitness}"


@dataclass
class Phylogeny:
    population_size: int
    input_size: int
    output_size: int
    fitness_function: tp.Callable[[Genome], float]
    weight_mutation_rate: float = 0.8
    weight_perturb_rate: float = 0.9
    link_mutate_rate: float = 0.05
    node_mutate_rate: float = 0.03
    disable_on_inherit_rate: float = 0.75
    inter_species_mating_rate: float = 0.001
    crossover_rate: float = 0.75
    survival_percentile: float = 0.3
    compatibility_scale_1: float = 1.0
    compatibility_scale_2: float = 1.0
    compatibility_scale_3: float = 0.4
    compatibility_threshold: float = 3.0
    max_generations: int = 100
    activation: tp.Callable[[float], float] = Activations.neat_sigmoid
    species: tp.List[Species] = None
    species_count: tp.ClassVar[int] = 0

    def __post_init__(self):
        initial_population = Genome.genesis(
            self.population_size,
            self.input_size,
            self.output_size,
            activation=self.activation,
            fitness_function=self.fitness_function,
        )

        self.species = [Species(id=0, members=initial_population)]
        self.species[0].reorganize()

    def speciate(self, new_generation: tp.List[Genome]):
        for species in self.species:
            species.reset_members()
        max_distance = 0
        for genome in new_generation:
            genome.evaluate_fitness()
            for species in self.species:
                d = species.compatibility(
                    genome,
                    self.compatibility_scale_1,
                    self.compatibility_scale_2,
                    self.compatibility_scale_3,
                )
                max_distance = max(max_distance, d)
                if d < self.compatibility_threshold:
                    species.members.append(genome)
                    break
            else:
                self.species_count += 1
                self.species.append(
                    Species(
                        id=self.species_count, members=[genome], _representative=genome
                    )
                )
        logger.info(f"Max Distance: {max_distance}")
        self.species = [s for s in self.species if s.size > 0]
        for species in self.species:
            species.reorganize()

    def mutate(self, genome: Genome) -> None:
        if np.random.rand() < self.weight_mutation_rate:
            genome.mutate_perturb_weights(perturb_rate=self.weight_perturb_rate)
        link_added = False
        if np.random.rand() < self.link_mutate_rate:
            link_added = genome.mutate_add_connection()
        if not link_added and np.random.rand() < self.node_mutate_rate:
            genome.mutate_add_node()

    def evolve(self, log: bool = False):
        for generation in range(1, self.max_generations):
            logger.info(f"GIN @ {generation} : {ConnectionGene.current_innovation}")

            new_population = []
            total_fitness = sum(s.adjusted_fitness for s in self.species)

            for species in self.species:
                fitness_rate = species.adjusted_fitness / total_fitness
                size = max(2, int(fitness_rate * self.population_size))
                new_population.append(species.champion)
                size -= 1
                survivor_index = max(
                    2, int(self.survival_percentile * len(species.members))
                )
                survivors = species.members[:survivor_index]
                survivor_fitness = sum(g.fitness for g in survivors)
                survivor_prob = [
                    (
                        g.fitness / survivor_fitness
                        if survivor_fitness > 0
                        else 1 / len(survivors)
                    )
                    for g in survivors
                ]
                crossovers = int(self.crossover_rate * size) if species.size > 1 else 0
                replicants = size - crossovers

                for _ in range(crossovers):
                    parent_1 = np.random.choice(survivors, p=survivor_prob)
                    parent_2 = np.random.choice(survivors, p=survivor_prob)
                    child = Genome.crossover(
                        parent_1,
                        parent_2,
                        disable_probability=self.disable_on_inherit_rate,
                    )
                    self.mutate(child)
                    new_population.append(child)
                for _ in range(replicants):
                    child: Genome = np.random.choice(survivors)
                    child = child.clone()
                    self.mutate(child)
                    new_population.append(child)

            self.speciate(new_population)

            if log:
                logger.info(f"Generation {generation}")
                logger.info(f"Species Count: {len(self.species)}")
                logger.info(
                    f"Best Genome Fitness: {max(p.fitness for p in new_population)}"
                )
                logger.info(f"Best Genome : {self.species[0].champion} ")
                print("")
        return new_population
