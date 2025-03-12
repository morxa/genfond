import logging
from typing import Collection, Mapping, Optional

from pddl.core import Domain, Problem

from .feature_generator import FeaturePool, get_goal_augmented_state
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
        log.debug(f'generated features: {", ".join(self.features.keys())}')

    def evaluate_feature(self, feature: str, problem: Problem, state: State) -> bool:
        log.warning("NOT IMPLEMENTED YET")
        return False

    def evaluate_concept(self, concept: str, problem: Problem, state: State) -> set[str]:
        return set()

    def evaluate_role(self, role: str, problem: Problem, state: State) -> set[tuple[str, str]]:
        return set()
