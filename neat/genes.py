from collections import defaultdict
from enum import Enum
import numpy as np
import dataclasses
from dataclasses import dataclass
import typing as tp
import itertools
from neat.activations import Activations
from neat.init_schemes import InitSchemes


class NodeType(Enum):
    SENSOR = 0
    HIDDEN = 1
    OUTPUT = 2


@dataclass
class NodeGene:
    OUTPUT_LEVEL: tp.ClassVar[int] = -1
    INPUT_LEVEL: tp.ClassVar[int] = 0

    id: int
    node_type: NodeType
    layer_level: int

    def __hash__(self) -> int:
        return hash(self.id)

    def __lt__(self, value: "NodeGene") -> bool:
        return self.id < value.id

    def __gt__(self, value: "NodeGene") -> bool:
        return self.id > value.id


@dataclass
class ConnectionGene:
    in_node: NodeGene
    out_node: NodeGene
    weight: float
    enabled: bool
    # innovation: int

    current_innovation: tp.ClassVar[int] = 0
    innovation_history: tp.ClassVar[tp.Dict[tp.Tuple[int], int]] = {}

    def __post_init__(self):
        cls = self.__class__
        if (self.in_node.id, self.out_node.id) in cls.innovation_history:
            self.innovation = cls.innovation_history[
                (self.in_node.id, self.out_node.id)
            ]
        else:
            cls.current_innovation += 1
            self.innovation = cls.current_innovation
            cls.innovation_history[(self.in_node.id, self.out_node.id)] = (
                self.innovation
            )


@dataclass
class Genome:
    # Genotype
    inputs: tp.Sequence[NodeGene]
    outputs: tp.Sequence[NodeGene]
    hidden: tp.List[NodeGene]
    connections: tp.List[ConnectionGene]
    activation: tp.Callable
    output_activation: tp.Union[tp.Callable, None] = None
    # Phenotype
    previous_activation: tp.Dict[int, float] = dataclasses.field(default_factory=dict)
    phenotype: tp.Dict[
        int, tp.List[tp.Tuple[NodeGene, tp.Sequence[ConnectionGene]]]
    ] = dataclasses.field(default_factory=dict)
    max_layer: int = 0
    species_id: int = 0
    fitness_function: tp.Callable[["Genome"], float] = None

    def __post_init__(self):
        self.max_nodes: int = len(self.inputs) + len(self.outputs) + len(self.hidden)
        self._fitness: float = 0
        self.build_graph()

    @classmethod
    def genesis(
        cls,
        population: int,
        input_nodes: int,
        output_nodes: int,
        init_scheme: tp.Callable = InitSchemes.uniform,
        activation: tp.Callable = Activations.neat_sigmoid,
        fitness_function: tp.Callable[["Genome"], float] = None,
    ) -> tp.List["Genome"]:

        # add bias
        input_nodes += 1

        population_out = []

        for _ in range(population):
            inputs = []
            outputs = []
            connections = []
            for input in range(input_nodes):
                inputs.append(NodeGene(input, NodeType.SENSOR, NodeGene.INPUT_LEVEL))
            for output in range(input_nodes, input_nodes + output_nodes):
                outputs.append(NodeGene(output, NodeType.OUTPUT, NodeGene.OUTPUT_LEVEL))

            for input in range(input_nodes):
                for output in range(output_nodes):
                    connections.append(
                        ConnectionGene(
                            inputs[input],
                            outputs[output],
                            init_scheme(input_nodes, output_nodes),
                            True,
                            # input * output_nodes + output + 1,
                        )
                    )
            population_out.append(
                Genome(
                    inputs,
                    outputs,
                    [],
                    connections,
                    activation=activation,
                    fitness_function=fitness_function,
                )
            )
            if fitness_function is not None:
                population_out[-1].evaluate_fitness()
        return population_out

    def evaluate_fitness(self) -> None:
        self._fitness = self.fitness_function(self)

    @property
    def fitness(self) -> float:
        return max(0, self._fitness)

    @staticmethod
    def distance(
        genome1: "Genome", genome2: "Genome", c1: float, c2: float, c3: float
    ) -> float:
        # Matching
        innovations1 = {
            connection.innovation: connection for connection in genome1.connections
        }
        innovations2 = {
            connection.innovation: connection for connection in genome2.connections
        }
        matching = innovations1.keys() & innovations2.keys()

        max1 = max(innovations1.keys())
        max2 = max(innovations2.keys())
        n = max(max1, max2)

        if n <= 20:
            n = 1

        disjoint = (innovations1.keys() | innovations2.keys()) - matching

        n_disjoint = len([d for d in disjoint if d <= min(max1, max2)])
        n_excess = len(disjoint) - n_disjoint
        weight_diff = 0

        for m in matching:
            weight_diff += abs(innovations1[m].weight - innovations2[m].weight)
        weight_diff /= len(matching) if matching else 1

        return c1 * n_disjoint / n + c2 * n_excess / n + c3 * weight_diff

    @staticmethod
    def crossover(
        parent1: "Genome",
        parent2: "Genome",
        disable_probability: float = 0.1,
    ) -> "Genome":
        # Matching
        innovations1 = {
            connection.innovation: connection for connection in parent1.connections
        }
        innovations2 = {
            connection.innovation: connection for connection in parent2.connections
        }
        matching = innovations1.keys() & innovations2.keys()
        disjoint = (
            innovations1.keys() - innovations2.keys()
            if parent1.fitness > parent2.fitness
            else innovations2.keys() - innovations1.keys()
        )
        disjoint_source = (
            innovations1 if parent1.fitness > parent2.fitness else innovations2
        )

        # Offspring setup
        inputs = parent1.inputs
        outputs = parent1.outputs
        hidden = []
        connections = []
        # max_layer = -1

        seen = set()

        # Crossover matching
        for innovation in matching:
            source = innovations1 if np.random.random() < 0.5 else innovations2
            inheritance = source[innovation]
            enabled = inheritance.enabled
            if (
                not innovations1[innovation].enabled
                or not innovations2[innovation].enabled
            ):
                enabled = np.random.random() > disable_probability

            connections.append(dataclasses.replace(inheritance, enabled=enabled))

            for node in (inheritance.in_node, inheritance.out_node):
                if node.id not in seen:
                    seen.add(node.id)
                    if node.node_type == NodeType.HIDDEN:
                        # max_layer = max(max_layer, node.layer_level)
                        hidden.append(node)
        # Crossover disjoint
        for innovation in disjoint:
            inheritance = disjoint_source[innovation]

            connections.append(dataclasses.replace(inheritance))
            for node in (
                disjoint_source[innovation].in_node,
                disjoint_source[innovation].out_node,
            ):
                if node.id not in seen:
                    seen.add(node.id)
                    if node.node_type == NodeType.HIDDEN:
                        # max_layer = max(max_layer, node.layer_level)
                        hidden.append(node)

        return Genome(
            inputs,
            outputs,
            hidden,
            connections,
            parent1.activation,
            parent1.output_activation,
            fitness_function=parent1.fitness_function,
        )

    def clone(self) -> "Genome":
        return Genome(
            self.inputs,
            self.outputs,
            [dataclasses.replace(x) for x in self.hidden],
            [dataclasses.replace(x) for x in self.connections],
            activation=self.activation,
            output_activation=self.output_activation,
            fitness_function=self.fitness_function,
        )

    def mutate_perturb_weights(
        self,
        uniform_draws: tp.Union[np.ndarray, None] = None,
        uniform_perturbs: tp.Union[np.ndarray, None] = None,
        perturb_rate: float = 0.9,
    ):
        uniform_draws = (
            np.random.uniform(0, 1, len(self.connections))
            if uniform_draws is None
            else uniform_draws
        )
        uniform_perturbs = (
            np.random.uniform(-1, 1, len(self.connections))
            if uniform_perturbs is None
            else uniform_perturbs
        )

        assert len(uniform_draws) == len(self.connections)
        assert len(uniform_perturbs) == len(self.connections)

        for idx, connection in enumerate(self.connections):
            if uniform_draws[idx] < perturb_rate:
                connection.weight += uniform_perturbs[idx]
            else:
                connection.weight = uniform_perturbs[idx]

    def mutate_add_connection(self):
        current_connections = set(
            (connection.in_node, connection.out_node) for connection in self.connections
        )
        possible_connections = list(
            filter(
                lambda x: (x[0], x[1]) not in current_connections and x[0] != x[1],
                itertools.product(
                    self.inputs + self.hidden + self.outputs, self.hidden + self.outputs
                ),
            )
        )

        if not possible_connections:
            return False

        connection = possible_connections[
            np.random.choice(range(len(possible_connections)))
        ]
        self.connections.append(
            ConnectionGene(
                connection[0],
                connection[1],
                np.random.uniform(-1, 1),
                True,
                # self.next_innovation(),
            )
        )
        return True

    def mutate_add_node(self):
        connection: ConnectionGene = np.random.choice(self.connections)

        while connection.in_node.layer_level == connection.out_node.layer_level:
            connection: ConnectionGene = np.random.choice(self.connections)

        connection.enabled = False

        if connection.in_node.layer_level == NodeGene.OUTPUT_LEVEL:
            new_node_level = connection.out_node.layer_level + 1
        elif connection.out_node.layer_level == NodeGene.OUTPUT_LEVEL:
            new_node_level = connection.in_node.layer_level + 1
        elif connection.in_node.layer_level > connection.out_node.layer_level:
            new_node_level = connection.out_node.layer_level + 1
            if new_node_level == connection.in_node.layer_level:
                connection.in_node.layer_level += 1
        else:
            new_node_level = connection.in_node.layer_level + 1
            if new_node_level == connection.out_node.layer_level:
                connection.out_node.layer_level += 1

        new_node = NodeGene(
            self.max_nodes,
            NodeType.HIDDEN,
            new_node_level,
        )
        self.max_nodes += 1
        self.hidden.append(new_node)

        self.connections.append(
            ConnectionGene(
                connection.in_node,
                new_node,
                1.0,
                True,
                # self.next_innovation(),
            )
        )

        self.connections.append(
            ConnectionGene(
                new_node,
                connection.out_node,
                connection.weight,
                True,
                # self.next_innovation(),
            )
        )

    def build_graph(self):
        layers = defaultdict(list)
        self.max_layer = 0

        connections = sorted(
            filter(lambda x: x.enabled, self.connections), key=lambda x: x.out_node
        )

        self.max_layer = 0
        for in_node, edges in itertools.groupby(connections, key=lambda x: x.out_node):
            self.max_layer = max(self.max_layer, in_node.layer_level)
            layers[in_node.layer_level].append((in_node, list(edges)))

        self.phenotype = layers
        self.previous_activation = {}

    def clear_memory(self):
        self.previous_activation = {}

    def forward(self, inputs: tp.Sequence, verbose=False) -> tp.Sequence[float]:
        # last element should be the bias node
        if len(inputs) != len(self.inputs):
            raise ValueError("Input length does not match genome input length")
        if not self.phenotype:
            self.build_graph()

        # create a cache for the current state
        current = {}

        # set the input values
        for i, input in enumerate(inputs):
            current[i] = input
        # run through hidden layers
        for l in range(1, self.max_layer + 1):
            if verbose:
                print(f"Layer Level: {l}")
            for in_node, edges in self.phenotype[l]:
                if verbose:
                    print(f"Node: {in_node.id}")
                current_activation = 0
                for edge in edges:
                    # Recurrent edge
                    if (
                        edge.in_node.layer_level >= l
                        or edge.in_node.layer_level == NodeGene.OUTPUT_LEVEL
                    ):
                        current_activation += (
                            edge.weight
                            * self.previous_activation.get(edge.in_node.id, 0)
                        )
                    else:
                        current_activation += edge.weight * current.get(
                            edge.in_node.id, 0
                        )
                    if verbose:
                        print(
                            f"Edge: {edge.in_node.id} -> {edge.out_node.id} : Activation {current_activation}"
                        )

                current[in_node.id] = self.activation(current_activation)
        # output layer
        for output, edges in self.phenotype[NodeGene.OUTPUT_LEVEL]:
            if verbose:
                print(f"Node: {output.id}")
            current_activation = 0
            for edge in edges:
                current_activation += edge.weight * current.get(edge.in_node.id, 0)
                if verbose:
                    print(
                        f"Edge: {edge.in_node.id} -> {edge.out_node.id} : Activation {current_activation}"
                    )

            current[output.id] = (
                self.output_activation(current_activation)
                if self.output_activation is not None
                else current_activation
            )
        # persist state to use for recurrent connections
        self.previous_activation = current

        return tuple(current.get(output.id, 0) for output in self.outputs)

    def __str__(self):
        out = f"Max Depth : {self.max_layer}\n"
        out += f"Input : {[f.id for f in self.inputs]}\n"
        out += f"Hidden : {[(f.id, f.layer_level) for f in self.hidden]}\n"
        out += f"Output : {[f.id for f in self.outputs]}\n"
        out += f"Connections : \n"
        for f in self.connections:
            out += f"{f.innovation}) {f.in_node.id} -> {f.out_node.id} : {f.enabled} @ {f.weight}\n"
        out += f"Fitness {self.fitness}: \n"

        return out

    def __repr__(self) -> str:
        return self.__str__()
