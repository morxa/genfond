import logging
from dataclasses import dataclass
from typing import Collection, Mapping, Optional

import numpy as np
from pddl.core import Domain, Problem
from pddl.logic.functions import EqualTo
from pddl.logic.predicates import Predicate
from pyvis.network import Network

from wlplan.data import Dataset, ProblemStates
from wlplan.feature_generation import Features, get_feature_generator
from wlplan.graph import to_networkx
from wlplan.planning import Atom
from wlplan.planning import Domain as WlDomain
from wlplan.planning import Problem as WlProblem
from wlplan.planning import State as WlState
from wlplan.planning import to_wlplan_domain, to_wlplan_problem

from .feature_generator import FeaturePool
from .state_space_generator import State, state_to_string

log = logging.getLogger("genfond.feature_generation_wlplan")


def get_sub_index_map(n_cat_features: int, n_con_features: int) -> dict[tuple[int, int], int]:
    i = n_cat_features + n_con_features
    ret = {}
    for c1 in range(n_con_features):
        for c2 in range(n_con_features):
            if c1 == c2:
                continue
            ret[(c1, c2)] = i
            i += 1
    return ret


def to_wlplan_state(state: State, wlplan_domain: WlDomain, wlplan_problem: WlProblem) -> WlState:
    predicate_to_id = {p.name: p for p in wlplan_domain.predicates}
    fluent_to_id = wlplan_problem.fluent_name_to_id
    atoms = []
    fluents = [0 for _ in range(len(fluent_to_id))]
    for k in state:
        if isinstance(k, Predicate):
            pred = predicate_to_id[k.name]
            terms = [o.name for o in k.terms]
            atom = Atom(pred, terms)
            atoms.append(atom)
        elif isinstance(k, EqualTo):
            fluent = str(k.operands[0])
            value = k.operands[1].value
            toks = fluent[1:-1].split(" ")
            function = toks[0]
            terms = toks[1:]
            fluent = f"{function}({', '.join(terms)})"
            fluents[fluent_to_id[fluent]] = value
        else:
            raise TypeError(f"{k}: unknown type for numeric state")
    return WlState(atoms, fluents)


@dataclass
class WlFeature:
    def __init__(self, id: int, name: str, complexity: int):
        self.id = id
        self.name = name
        self.complexity = complexity

    def compute_complexity(self) -> int:
        return self.complexity


class WlPlanFeaturePool(FeaturePool):
    def __init__(
        self,
        domain: Domain,
        problems: Collection[Problem],
        config: Mapping,
        max_complexity: Optional[int] = None,
        selected_states: Optional[dict[str, set[State]]] = None,
    ):
        super().__init__(domain, problems, config, max_complexity, selected_states)
        self.features: dict[str, WlFeature] = {}

        wlplan_domain = to_wlplan_domain(pddl_domain=domain)

        fg = get_feature_generator(
            domain=wlplan_domain,
            graph_representation="nilg",
            feature_algorithm="ccwl-a",
            iterations=max_complexity,
        )

        problem_states = []
        self.problem_state_to_row: dict[tuple[str, str], int] = {}
        self.row_to_problem_state: dict[int, tuple[str, str]] = {}

        for problem in problems:
            wlplan_problem = to_wlplan_problem(pddl_domain=domain, pddl_problem=problem)

            wlplan_states = []
            for state, _ in self.state_graphs[problem.name].nodes.items():
                wlplan_states.append(to_wlplan_state(state, wlplan_domain, wlplan_problem))
                ps_key = problem.name, state_to_string(state)
                row = len(self.problem_state_to_row)
                self.problem_state_to_row[ps_key] = row
                self.row_to_problem_state[row] = ps_key
            problem_states.append(ProblemStates(problem=wlplan_problem, states=wlplan_states))

        dataset = Dataset(domain=wlplan_domain, data=problem_states)
        fg.collect(dataset)

        self.X = fg.embed(dataset)
        self.X = np.array(self.X).astype(int)
        layer_to_colours = fg.get_layer_to_colours()
        colour_to_layer = {colour: i for i in range(len(layer_to_colours)) for colour in layer_to_colours[i]}

        # partition features
        n_cat_features = len(colour_to_layer)
        n_con_features = len(colour_to_layer)
        n_sub_features = n_con_features * (n_con_features - 1)

        # for c in range(n_cat_features):
        #     feature = WlFeature(id=c, name=f"n_{c}_cat", complexity=colour_to_layer[c])
        #     self.features[feature.name] = feature

        for c in range(n_con_features):
            feature = WlFeature(id=n_cat_features + c, name=f"n_{c}_con", complexity=colour_to_layer[c])
            self.features[feature.name] = feature

        sub_index_map = get_sub_index_map(n_cat_features, n_con_features)
        for c1 in range(n_con_features):
            for c2 in range(n_con_features):
                if c1 == c2:
                    continue
                colour = max(colour_to_layer[c1], colour_to_layer[c2])
                feature = WlFeature(id=sub_index_map[(c1, c2)], name=f"n_{c1}-{c2}_sub", complexity=colour)
                self.features[feature.name] = feature

        fg.save(self.get_save_file())

        log.debug(f'generated features: {", ".join(self.features.keys())}')

        # self._debug_feature_generator(fg, dataset)

    def _debug_feature_generator(self, fg: Features, dataset: Dataset) -> None:
        graphs = fg.convert_to_graphs(dataset)
        # for i in [0, 1]:
        for i in range(len(graphs)):
            graph = graphs[i]
            # graph.dump()
            G = to_networkx(graph)
            net = Network(height="1080px", notebook=False)

            # Add nodes
            for node, node_attrs in G.nodes(data=True):
                colour = node_attrs["colour"]
                label = f"{colour}\n{node}"
                net.add_node(node, label=label)

            # Add edges
            for source, target, edge_attrs in G.edges(data=True):
                net.add_edge(source, target)

            net.from_nx(G)
            net.set_options('{"edges":{"smooth":false},"physics":{"enabled":false}}')
            save_file = f"graph{i}.html"
            net.save_graph(save_file)
            log.debug(f"Drew graph to {save_file}")
            state = self.row_to_problem_state[i][1]
            log.debug(f"{state=}")
            n = self.X.shape[1]
            top = self.X[i][: n // 2]
            bot = self.X[i][n // 2 :]
            assert len(top) == len(bot)
            log.debug(f"feature_vec={' '.join(map(str, top))}")
            log.debug(f"            {' '.join(map(str, bot))}")
        # breakpoint()

        # fg.save("features.model")

    def get_augmented_state(self, problem: Problem, state: State) -> State:
        return state

    def evaluate_feature(self, feature: str, problem: Problem, state: State) -> int:
        ps_key = (problem.name, state_to_string(state))
        row = self.problem_state_to_row[ps_key]
        col = self.features[feature].id
        return int(self.X[row][col])

    def evaluate_concept(self, concept: str, problem: Problem, state: State) -> set[str]:
        return set()

    def evaluate_role(self, role: str, problem: Problem, state: State) -> set[tuple[str, str]]:
        return set()

    def get_save_file(self) -> str:
        return "wlplan_features.model"
