import logging
from typing import Collection, Mapping, Optional

import dlplan.core
import dlplan.generator as dlplan_gen
from dlplan.core import (
    Atom,
    Boolean,
    Concept,
    InstanceInfo,
    Numerical,
    Role,
    SyntacticElementFactory,
    VocabularyInfo,
)
from pddl.core import Domain, Problem
from pddl.logic import Predicate

from .feature_generator import FeaturePool, get_goal_augmented_state
from .ground import ground_domain_predicates
from .state_space_generator import State

type Feature = Boolean | Numerical

log = logging.getLogger("genfond.feature_generation_dlplan")


def construct_vocabulary_info(domain: Domain, config: Mapping) -> VocabularyInfo:
    vocabulary = VocabularyInfo()
    for predicate in domain.predicates:
        assert f"{predicate.name}_G" not in [p.name for p in domain.predicates]
    for predicate in domain.predicates:
        # TODO some predicates may be static.
        vocabulary.add_predicate(predicate.name, predicate.arity)
        vocabulary.add_predicate(f"{predicate.name}_G", predicate.arity)
    for constant in domain.constants:
        vocabulary.add_constant(constant.name)
    return vocabulary


def construct_instance_info(
    vocabulary: VocabularyInfo, domain: Domain, problem: Problem, problem_id: int, config: Mapping
) -> tuple[InstanceInfo, dict[Predicate, Atom]]:
    instance = InstanceInfo(problem_id, vocabulary)
    map = dict()
    for object in problem.objects:
        instance.add_object(object.name)
    for predicate in ground_domain_predicates(domain, problem):
        map[predicate] = instance.add_atom(predicate.name, [str(t) for t in predicate.terms])
        goal_predicate = Predicate(f"{predicate.name}_G", *predicate.terms)
        map[goal_predicate] = instance.add_atom(goal_predicate.name, [str(t) for t in predicate.terms])
    return instance, map


class DlPlanFeaturePool(FeaturePool):

    def __init__(
        self,
        domain: Domain,
        problems: Collection[Problem],
        config: Mapping,
        max_complexity: Optional[int] = None,
        all_generators: bool = False,
        selected_states: Optional[dict[str, set[State]]] = None,
    ):
        super().__init__(domain, problems, config, selected_states)
        self.instances: dict[str, InstanceInfo] = dict()
        self.mappings: dict[str, dict[Predicate, Atom]] = dict()
        self.max_complexity = max_complexity or config["max_complexity"]
        vocabulary = construct_vocabulary_info(domain, config)
        log.debug(f"Constructed vocabulary: {vocabulary}")
        for problem in problems:
            instance, mapping = construct_instance_info(
                vocabulary,
                domain,
                problem,
                self.problem_name_to_id[problem.name],
                config,
            )
            self.instances[problem.name] = instance
            self.mappings[problem.name] = mapping
        dlplan_states: list[dlplan.core.State] = []
        for state_graph in self.state_graphs.values():
            for node in state_graph.nodes.values():
                dlplan_states.append(
                    self.state_to_dlplan_state(
                        state_graph.problem, get_goal_augmented_state(state_graph.problem, node.state)
                    )
                )
        factory = SyntacticElementFactory(vocabulary)
        if config.get("preset_features", None):
            booleans = [factory.parse_boolean(f) for f in config["preset_features"].get("booleans", [])]
            numericals = [factory.parse_numerical(f) for f in config["preset_features"].get("numericals", [])]
            concepts = [factory.parse_concept(c) for c in config["preset_features"].get("concepts", [])]
            roles = [factory.parse_role(r) for r in config["preset_features"].get("roles", [])]
        else:
            if all_generators:
                feature_generator_kwargs = config["unrestricted_feature_generator"]
            else:
                feature_generator_kwargs = config["feature_generator"]
            booleans, numericals, concepts, roles = dlplan_gen.generate_features(
                factory,
                dlplan_states,
                *5 * [self.max_complexity],
                3600,
                10000,
                **feature_generator_kwargs,
            )
        self.features: dict[str, Feature] = {}
        self.concepts: dict[str, Concept] = {}
        self.roles: dict[str, Role] = {}
        if config["include_boolean_features"]:
            for feature in booleans:
                self.features[str(feature)] = feature
        if config["include_numerical_features"]:
            for feature in numericals:
                self.features[str(feature)] = feature
        if config["include_concepts"]:
            self.concepts = {str(concept): concept for concept in concepts}
        if config["include_roles"]:
            self.roles = {str(role): role for role in roles}
        log.debug(f'generated concepts: {", ".join(self.concepts.keys())}')
        log.debug(f'generated roles: {", ".join(self.roles.keys())}')
        log.debug(f'generated features: {", ".join(self.features.keys())}')

    def state_to_dlplan_state(self, problem: Problem, state: State, state_id: int = -1) -> dlplan.core.State:
        return dlplan.core.State(
            state_id,
            self.instances[problem.name],
            [self.mappings[problem.name][fact] for fact in state],
        )

    def evaluate_feature(self, feature: str, problem: Problem, state: State) -> bool:
        feature = feature.strip('"')
        return self.features[feature].evaluate(
            self.state_to_dlplan_state(problem, get_goal_augmented_state(problem, state))
        )

    def evaluate_concept(self, concept: str, problem: Problem, state: State) -> set[str]:
        concept = concept.strip('"')
        return set(
            [
                self.obj_id_to_obj(problem, id)
                for id in self.concepts[concept]
                .evaluate(self.state_to_dlplan_state(problem, get_goal_augmented_state(problem, state)))
                .to_sorted_vector()
            ]
        )

    def evaluate_role(self, role: str, problem: Problem, state: State) -> set[tuple[str, str]]:
        role = role.strip('"')
        return set(
            [
                (self.obj_id_to_obj(problem, id1), self.obj_id_to_obj(problem, id2))
                for id1, id2 in self.roles[role]
                .evaluate(self.state_to_dlplan_state(problem, get_goal_augmented_state(problem, state)))
                .to_sorted_vector()
            ]
        )

    def obj_id_to_obj(self, problem: Problem, obj_id: int) -> str:
        for obj in self.instances[problem.name].get_objects():
            if obj_id == obj.get_index():
                return obj.get_name()
        raise KeyError(f"Cannot find object with id {obj_id}")
