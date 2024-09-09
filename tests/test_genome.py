import math
import numpy as np
import pytest
from neat.genes import ConnectionGene, Genome, NodeGene, NodeType
from neat.init_schemes import InitSchemes
from neat.activations import Activations
import typing as tp

# Setup fixtures
@pytest.fixture(scope="function")
def genome1() -> Genome:
    inputs = [
        NodeGene(0, NodeType.SENSOR, 0),
        NodeGene(1, NodeType.SENSOR, 0),
        NodeGene(2, NodeType.SENSOR, 0),
    ]
    outputs = [
        NodeGene(3, NodeType.OUTPUT, -1),
    ]

    hidden = [
        NodeGene(4, NodeType.HIDDEN, 1),
    ]

    return Genome(
        inputs=inputs,
        outputs=outputs,
        hidden=hidden,
        connections=[
            ConnectionGene(inputs[0], outputs[0], 1.0, True, 1),
            ConnectionGene(inputs[1], outputs[0], 1.0, False, 2),
            ConnectionGene(inputs[2], outputs[0], 1.0, True, 3),
            ConnectionGene(inputs[1], hidden[0], 1.0, True, 4),
            ConnectionGene(hidden[0], outputs[0], 1.0, True, 5),
            ConnectionGene(inputs[0], hidden[0], 1.0, True, 6),
            ConnectionGene(outputs[0], hidden[0], 1.0, True, 7),
        ],
        activation=Activations.neaty_sigmoid,
    )


@pytest.fixture(scope="function")
def genome2() -> Genome:
    inputs = [
        NodeGene(0, NodeType.SENSOR, 0),
        NodeGene(1, NodeType.SENSOR, 0),
        NodeGene(2, NodeType.SENSOR, 0),
    ]
    outputs = [
        NodeGene(3, NodeType.OUTPUT, -1),
    ]

    hidden = [
        NodeGene(4, NodeType.HIDDEN, 1),
    ]

    return Genome(
        inputs=inputs,
        outputs=outputs,
        hidden=hidden,
        connections=[
            ConnectionGene(inputs[0], outputs[0], 1.0, True, 1),
            ConnectionGene(inputs[1], outputs[0], 1.0, False, 2),
            ConnectionGene(inputs[2], outputs[0], 1.0, True, 3),
            ConnectionGene(inputs[1], hidden[0], 1.0, True, 4),
            ConnectionGene(hidden[0], outputs[0], 1.0, True, 5),
            ConnectionGene(inputs[0], hidden[0], 1.0, True, 6),
        ],
        activation=Activations.neaty_sigmoid,
    )


@pytest.fixture(scope="function")
def crossover_parents() -> tp.Tuple[Genome, Genome]:
    inputs = [
        NodeGene(0, NodeType.SENSOR, 0),
        NodeGene(1, NodeType.SENSOR, 0),
        NodeGene(2, NodeType.SENSOR, 0),
    ]
    outputs = [
        NodeGene(3, NodeType.OUTPUT, -1),
    ]

    hidden = [
        NodeGene(4, NodeType.HIDDEN, 1),
    ]

    genome1 = Genome(
        inputs=inputs,
        outputs=outputs,
        hidden=hidden,
        connections=[
            ConnectionGene(inputs[0], outputs[0], 1.0, True, 1),
            ConnectionGene(inputs[1], outputs[0], 1.0, False, 2),
            ConnectionGene(inputs[2], outputs[0], 1.0, True, 3),
            ConnectionGene(inputs[1], hidden[0], 1.0, True, 4),
            ConnectionGene(hidden[0], outputs[0], 1.0, True, 5),
            ConnectionGene(inputs[0], hidden[0], 1.0, True, 8),
        ],
        activation=Activations.neaty_sigmoid,
    )

    hidden2 = hidden + [NodeGene(5, NodeType.HIDDEN, 2)]

    genome2 = Genome(
        inputs=inputs,
        outputs=outputs,
        hidden=hidden + [NodeGene(5, NodeType.HIDDEN, 2)],
        connections=[
            ConnectionGene(inputs[0], outputs[0], 1.0, True, 1),
            ConnectionGene(inputs[1], outputs[0], 1.0, False, 2),
            ConnectionGene(inputs[2], outputs[0], 1.0, True, 3),
            ConnectionGene(inputs[1], hidden2[0], 1.0, True, 4),
            ConnectionGene(hidden2[0], outputs[0], 1.0, False, 5),
            ConnectionGene(hidden2[0], hidden2[1], 1.0, True, 6),
            ConnectionGene(hidden2[1], outputs[0], 1.0, True, 7),
            ConnectionGene(inputs[2], hidden2[0], 1.0, True, 9),
            ConnectionGene(inputs[0], hidden2[1], 1.0, True, 10),
        ],
        activation=Activations.neaty_sigmoid,
    )

    return genome1, genome2

# Begin tests
def test_genome_genesis():
    INPUTS = 2
    OUTPUTS = 2
    POPULATION = 2

    genome = Genome.genesis(POPULATION, INPUTS, OUTPUTS, InitSchemes.one)
    assert len(genome) == POPULATION

    for g in genome:
        assert len(g.inputs) == 3
        assert len(g.outputs) == 2
        assert len(g.connections) == 6

        for c in g.connections:
            assert c.enabled
            assert c.weight == 1.0


def test_genome_mutation_add_connection(genome1: Genome):
    inputs = [1.0, 3.0, 2.0]
    activation = genome1.activation
    initial_connections = len(genome1.connections)
    exp_out = activation(inputs[0] + inputs[1]) + inputs[0] + inputs[2]

    genome1.build_graph()
    assert math.isclose(genome1.forward(inputs)[0], exp_out)

    genome1.mutate_add_connection()
    genome1.build_graph()
    assert len(genome1.connections) == initial_connections + 1

    # Mutated genome should have a different output
    assert not math.isclose(genome1.forward(inputs)[0], exp_out, rel_tol=1e-15)


def test_genome_mutation_perturb_weights(genome1: Genome):
    inputs = [1.0, 3.0, 2.0]
    activation = genome1.activation
    initial_connections = len(genome1.connections)
    initial_hidden = len(genome1.hidden)
    exp_out = activation(inputs[0] + inputs[1]) + inputs[0] + inputs[2]

    genome1.build_graph()
    assert math.isclose(genome1.forward(inputs)[0], exp_out)

    genome1.mutate_perturb_weights()
    genome1.build_graph()

    assert len(genome1.connections) == initial_connections
    assert len(genome1.hidden) == initial_hidden
    # Mutated genome should have a different output
    assert not math.isclose(genome1.forward(inputs)[0], exp_out)


def test_genome_mutation_add_node(genome1: Genome):
    np.random.seed(9)
    inputs = [1.0, 3.0, 2.0]
    activation = genome1.activation
    initial_connections = len(genome1.connections)
    initial_hidden = len(genome1.hidden)
    initial_nodes = genome1.max_nodes
    initial_layers = genome1.max_layer

    exp_out = activation(inputs[0] + inputs[1]) + inputs[0] + inputs[2]

    genome1.build_graph()
    f0 = genome1.forward(inputs)[0]
    # Sanity Check
    assert math.isclose(f0, exp_out)

    # Mutations
    genome1.mutate_add_node()
    genome1.build_graph()
    f1 = genome1.forward(inputs)[0]
    f2 = genome1.forward(inputs)[0]

    assert len(genome1.connections) != initial_connections
    assert len(genome1.hidden) != initial_hidden
    assert genome1.max_nodes != initial_nodes
    assert genome1.max_layer != initial_layers
    assert f2 != f1


def test_phenotype_forward(genome1: Genome):
    inputs = [1.0, 3.0, 2.0]
    activation = genome1.activation
    exp_out = activation(inputs[0] + inputs[1]) + inputs[0] + inputs[2]
    genome1.build_graph()

    # Test single forward pass
    assert math.isclose(genome1.forward(inputs)[0], exp_out)

    # Test recurrent forward pass
    exp_out_rec = activation(inputs[0] + inputs[1] + exp_out) + inputs[0] + inputs[2]
    assert math.isclose(genome1.forward(inputs)[0], exp_out_rec)

    exp_out_rec = (
        activation(inputs[0] + inputs[1] + exp_out_rec) + inputs[0] + inputs[2]
    )
    assert math.isclose(genome1.forward(inputs)[0], exp_out_rec)

    genome1.clear_memory()
    # Test forward pass after clearing recurrent memory
    assert math.isclose(genome1.forward(inputs)[0], exp_out)


def test_genome_crossover(crossover_parents: tp.Tuple[Genome, Genome]):
    parent1, parent2 = crossover_parents
    inputs = [1.0, 3.0, 2.0]

    child1 = Genome.crossover(parent1, parent2, True)
    child2 = Genome.crossover(parent1, parent2, False)

    assert len(child1.connections) == 6
    assert len(child2.connections) == 9
    assert len(child1.hidden) == 1
    assert len(child2.hidden) == 2

    # Test forward pass
    child1.forward(inputs)
    child2.forward(inputs)


def test_genome_distance(crossover_parents: tp.Tuple[Genome, Genome]):
    parent1, parent2 = crossover_parents
    c1, c2, c3 = 0.5, 0.5, 0.5
    excess = 2
    disjoint = 3
    matching = 5

    parent2.mutate_perturb_weights(
        np.full(len(parent2.connections), 0.5),
        np.full(len(parent2.connections), 2),
        0.9,
    )

    d_expected = excess * c1 + disjoint * c2 + c3 * 2

    assert Genome.distance(parent1, parent2, c1, c2, c3) == d_expected
