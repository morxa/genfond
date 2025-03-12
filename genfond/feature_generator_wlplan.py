import logging
from typing import Collection, Mapping, Optional

from pddl.core import Domain, Problem
from pddl.logic.functions import EqualTo
from pddl.logic.predicates import Predicate

from wlplan.data import Dataset, ProblemStates
from wlplan.feature_generation import get_feature_generator
from wlplan.planning import Atom, to_wlplan_domain, to_wlplan_problem

from .feature_generator import FeaturePool
from .state_space_generator import State

log = logging.getLogger("genfond.feature_generation_wlplan")


class WlPlanFeaturePool(FeaturePool):

    def __init__(
        self,
        domain: Domain,
        problems: Collection[Problem],
        config: Mapping,
        iterations: int = 2,
        selected_states: Optional[dict[str, set[State]]] = None,
    ):
        super().__init__(domain, problems, config, selected_states)
        self.features: dict[str, int] = {}

        wlplan_domain = to_wlplan_domain(pddl_domain=domain)

        fg = get_feature_generator(
            domain=wlplan_domain,
            graph_representation="nilg",
            feature_algorithm="ccwl",
            iterations=iterations,
        )

        problem_states = []

        for problem in problems:
            wlplan_problem = to_wlplan_problem(pddl_domain=domain, pddl_problem=problem)
            predicate_to_id = {p.name: p for p in wlplan_domain.predicates}
            fluent_to_id = wlplan_problem.fluent_name_to_id

            wlplan_states = []
            for state, _ in self.state_graphs[problem.name].nodes.items():
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
                wlplan_states.append(State(atoms, fluents))
            problem_states.append(ProblemStates(problem=wlplan_problem, states=wlplan_states))

        dataset = Dataset(domain=wlplan_domain, data=problem_states)
        fg.collect(dataset)
        # TODO something about the collected features

        log.debug(f'generated features: {", ".join(self.features.keys())}')

    def evaluate_feature(self, feature: str, problem: Problem, state: State) -> bool:
        log.warning("NOT IMPLEMENTED YET")
        return False

    def evaluate_concept(self, concept: str, problem: Problem, state: State) -> set[str]:
        return set()

    def evaluate_role(self, role: str, problem: Problem, state: State) -> set[tuple[str, str]]:
        return set()
