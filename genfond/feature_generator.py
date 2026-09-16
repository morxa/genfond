import itertools
import logging
from typing import Collection, Mapping, MutableMapping, Optional, Tuple

import dlplan.core
import dlplan.generator as dlplan_gen
from dlplan.core import Atom, Boolean, InstanceInfo, Numerical, SyntacticElementFactory, VocabularyInfo
from pddl.action import Action
from pddl.core import Domain, Formula, Plan, Problem
from pddl.logic import Predicate

from genfond.state_space_vis import draw_state_graph

from .action_signatures import ActionSignature, iter_dist_set_facts
from .forced_labels import ActionOccurrence, ForcedLabels, compute_forced_labels
from .ground import ground, ground_domain_predicates
from .state_space_generator import (
    Alive,
    State,
    StateSpaceGraph,
    StateSpaceNode,
    apply_effects,
    check_formula,
    generate_state_space,
)

type Feature = Boolean | Numerical

log = logging.getLogger("genfond.feature_generation")

MAX_ACTION_PARAMETERS = 4


def get_aparam_predicate_name(i: int) -> str:
    return f"aparam{i}"


def get_aparam_predicate(i: int, param: str) -> Predicate:
    return Predicate(get_aparam_predicate_name(i), param)


def construct_vocabulary_info(domain: Domain, config: Mapping) -> VocabularyInfo:
    vocabulary = VocabularyInfo()
    for predicate in domain.predicates:
        assert f"{predicate.name}_g" not in [p.name for p in domain.predicates]
    for predicate in domain.predicates:
        # TODO some predicates may be static.
        vocabulary.add_predicate(predicate.name, predicate.arity)
        vocabulary.add_predicate(f"{predicate.name}_g", predicate.arity)
    max_arity = max([len(action.parameters) for action in domain.actions])
    if config["include_actions"] or config["include_action_params"]:
        for i in range(max_arity):
            # log.debug(f'Adding action parameter predicate {get_aparam_predicate_name(i)}')
            aparam_pred = get_aparam_predicate_name(i)
            assert aparam_pred not in [p.name for p in domain.predicates]
            vocabulary.add_predicate(aparam_pred, 1)
    for constant in domain.constants:
        vocabulary.add_constant(constant.name)
    return vocabulary


def _get_state_from_goal(goal_formula: Formula):
    states = apply_effects(set([frozenset()]), goal_formula)
    assert len(states) == 1, f"Goal formula must define a unique goal state, found {len(states)} states: {states}"
    state = next(iter(states))
    goal_state = {Predicate(f"{predicate.name}_g", *predicate.terms) for predicate in state}
    return goal_state


def construct_instance_info(
    vocabulary: VocabularyInfo, domain: Domain, problem: Problem, problem_id: int, config: Mapping
) -> tuple[InstanceInfo, dict[Predicate, Atom]]:
    instance = InstanceInfo(problem_id, vocabulary)
    map = dict()
    for object in problem.objects:
        instance.add_object(object.name)
    for predicate in ground_domain_predicates(domain, problem):
        map[predicate] = instance.add_atom(predicate.name, [str(t) for t in predicate.terms])
        goal_predicate = Predicate(f"{predicate.name}_g", *predicate.terms)
        map[goal_predicate] = instance.add_atom(goal_predicate.name, [str(t) for t in predicate.terms])
    if config["include_actions"]:
        for action in ground(domain, problem):
            for i, param in enumerate(action.parameters):
                map[get_aparam_predicate(i, param)] = instance.add_atom(get_aparam_predicate_name(i), [str(param)])
    if config["include_action_params"]:
        for action in ground(domain, problem):
            for i, param in enumerate(action.parameters):
                map[get_aparam_predicate(0, param)] = instance.add_atom(get_aparam_predicate_name(0), [str(param)])
    return instance, map


def get_goal_augmented_state(problem: Problem, state: State) -> State:
    return frozenset(state | _get_state_from_goal(problem.goal))


def get_param_augmented_state(problem: Problem, state: State, _, param: str) -> State:
    augmented_state = set(get_goal_augmented_state(problem, state))
    augmented_state.add(get_aparam_predicate(0, param))  # Always use 0 as we do not need the index
    # action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
    # log.debug(f'Generated augmented state (action={action_str}): [{", ".join([str(p) for p in augmented_state])}]')
    return frozenset(augmented_state)


def get_action_augmented_state(problem: Problem, state: State, config: Mapping, action=Optional[Action]) -> State:
    assert not config["include_actions"] or action, "Action must be provided when including actions"
    augmented_state = get_goal_augmented_state(problem, state)
    if config["include_actions"]:
        param_atoms = {get_aparam_predicate(i, action.parameters[i]) for i, _ in enumerate(action.parameters)}
        augmented_state |= param_atoms
    action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"' if action else "None"
    return frozenset(augmented_state)


def _parse_extra_features(
    factory: SyntacticElementFactory, extra_features_config: Mapping
) -> Tuple[list[Boolean], list[Numerical], list, list]:
    """Parse the booleans/numericals/concepts/roles of an `extra_features`-shaped config entry
    with `factory`. Unlike the synthesised pool, these go straight through `factory.parse_*`, so
    they are not subject to any complexity limit -- see the call sites in `FeaturePool.__init__`."""
    return (
        [factory.parse_boolean(f) for f in extra_features_config.get("booleans", [])],
        [factory.parse_numerical(f) for f in extra_features_config.get("numericals", [])],
        [factory.parse_concept(c) for c in extra_features_config.get("concepts", [])],
        [factory.parse_role(r) for r in extra_features_config.get("roles", [])],
    )


def _dedup_extend(elements: list, extra: list) -> list:
    """Append `extra` dlplan elements to `elements` in place, skipping any whose string
    representation already appears in `elements` (or earlier in `extra`). Returns the elements
    that were actually appended, in the same order."""
    seen = {str(element) for element in elements}
    added = []
    for element in extra:
        key = str(element)
        if key in seen:
            continue
        seen.add(key)
        elements.append(element)
        added.append(element)
    return added


def _log_added_extra_features(
    added_booleans: list, added_numericals: list, added_concepts: list, added_roles: list
) -> None:
    log.info(
        f"extra_features added {len(added_booleans)} boolean(s) "
        f"(complexities: {[b.compute_complexity() for b in added_booleans]}), "
        f"{len(added_numericals)} numerical(s) "
        f"(complexities: {[n.compute_complexity() for n in added_numericals]}), "
        f"{len(added_concepts)} concept(s) "
        f"(complexities: {[c.compute_complexity() for c in added_concepts]}), "
        f"{len(added_roles)} role(s) "
        f"(complexities: {[r.compute_complexity() for r in added_roles]})"
    )


class FeaturePool:

    def __init__(
        self,
        domain: Domain,
        problems: Collection[Problem],
        config: Mapping,
        max_complexity: Optional[int] = None,
        all_generators: bool = False,
        selected_states: Optional[Mapping[str, Collection[State]]] = None,
        plans: Optional[Mapping[str, Collection[Plan]]] = None,
        dead_states: Optional[Mapping[str, Collection[State]]] = None,
    ):
        assert len({problem.name for problem in problems}) == len(problems), "Problem names must be unique."
        self.domain = domain
        self.problems = {problem.name: problem for problem in problems}
        self.config = config
        self.problem_name_to_id = {problem.name: i for i, problem in enumerate(problems)}
        self.problem_id_to_name = {i: name for name, i in self.problem_name_to_id.items()}
        vocabulary = construct_vocabulary_info(domain, config)
        log.debug(f"Constructed vocabulary: {vocabulary}")
        self.states: dict[State, dlplan.core.State] = dict()
        self.node_id_to_state_ids: dict[tuple[int, int], State] = dict()
        self.node_id_to_action_aug_state_ids: dict[tuple[int, int], dict[Action, State]] = dict()
        self.node_id_to_param_aug_state_ids: dict[tuple[int, int], dict[Action, list[State]]] = dict()
        self.state_id_to_node: dict[State, list[StateSpaceNode]] = dict()
        self.state_graphs: MutableMapping[str, StateSpaceGraph] = dict()
        self._node_indices: dict[str, dict[int, StateSpaceNode]] = dict()
        self.instances: dict[str, InstanceInfo] = dict()
        self.mappings = dict()
        self.next_state_id = 0
        # Action signatures: see _emit_action_signatures. Only populated when the config asks
        # for them; the key is everything the datalog separation constraint can observe.
        self.signature_ids: dict[ActionSignature, int] = dict()
        # The asig/4 occurrences at alive non-goal states, collected while writing the per-node
        # facts and handed to forced_labels.compute_forced_labels once they are all known.
        self._occurrences: list[ActionOccurrence] = []
        # (instance, state, action) triples whose action starts an example plan suffix at that
        # state; only collected for the plan_label_heuristic.
        self._plan_actions: list[tuple[int, int, str]] = []
        self.forced_labels: Optional[ForcedLabels] = None
        # String keys of extra_features elements actually added to the pool (booleans and
        # numericals share self.features, so both land in _extra_feature_keys). Consulted in
        # to_clingo to apply extra_features_complexity, if set.
        self._extra_feature_keys: set[str] = set()
        self._extra_concept_keys: set[str] = set()
        self._extra_role_keys: set[str] = set()
        if not max_complexity:
            max_complexity = config["max_complexity"]
        for problem in problems:
            instance, mapping = construct_instance_info(
                vocabulary,
                domain,
                problem,
                self.problem_name_to_id[problem.name],
                config,
            )
            self.state_graphs[problem.name] = generate_state_space(
                domain,
                problem,
                selected_states=(selected_states.get(problem.name, None) if selected_states else None),
                plans=(plans.get(problem.name, None) if plans else None),
                frontier=config["frontier_expansion"],
                dead_states=(dead_states.get(problem.name, None) if dead_states else None),
            )
            if config["visualize_state_graphs"]:
                draw_state_graph(self.state_graphs[problem.name], f"{problem.name}_state_graph.png")
            self.instances[problem.name] = instance
            self.mappings[problem.name] = mapping
            for node in self.state_graphs[problem.name].nodes.values():
                if config["include_actions"]:
                    self.node_to_action_augmented_state(problem, node)
                elif config["include_action_params"]:
                    self.node_to_param_augmented_state(problem, node)
                if config["include_pristine_states"]:
                    self.node_to_state(problem, node)
        factory = SyntacticElementFactory(vocabulary)
        if config.get("preset_features", None):
            booleans = [factory.parse_boolean(f) for f in config["preset_features"].get("booleans", [])]
            numericals = [factory.parse_numerical(f) for f in config["preset_features"].get("numericals", [])]
            concepts = [factory.parse_concept(c) for c in config["preset_features"].get("concepts", [])]
            roles = [factory.parse_role(r) for r in config["preset_features"].get("roles", [])]
            if config.get("extra_features", None):
                # preset_features already replaces synthesis entirely, so there is no
                # generated pool to append after -- extra_features is simply merged in.
                extra_booleans, extra_numericals, extra_concepts, extra_roles = _parse_extra_features(
                    factory, config["extra_features"]
                )
                added_booleans = _dedup_extend(booleans, extra_booleans)
                added_numericals = _dedup_extend(numericals, extra_numericals)
                added_concepts = _dedup_extend(concepts, extra_concepts)
                added_roles = _dedup_extend(roles, extra_roles)
                _log_added_extra_features(added_booleans, added_numericals, added_concepts, added_roles)
                self._extra_feature_keys.update(str(b) for b in added_booleans)
                self._extra_feature_keys.update(str(n) for n in added_numericals)
                self._extra_concept_keys.update(str(c) for c in added_concepts)
                self._extra_role_keys.update(str(r) for r in added_roles)
        else:
            if all_generators:
                feature_generator_kwargs = config["unrestricted_feature_generator"]
            else:
                feature_generator_kwargs = config["feature_generator"]
            # Roles cost n*m^2 grounded values per state and concepts n*m, against n for plain
            # features, so at high complexity roles dominate the ground program (see
            # docs/role-caps-results.md). concept_complexity_offset/role_complexity_offset let a
            # round cap those two generators below the complexity used for booleans/numericals;
            # an offset of 0 (the default) reproduces the previous behaviour of five equal limits.
            concept_complexity_limit = max(1, max_complexity - config["concept_complexity_offset"])
            role_complexity_limit = max(1, max_complexity - config["role_complexity_offset"])
            log.info(
                f"Feature generation limits: complexity={max_complexity}, "
                f"concept={concept_complexity_limit}, role={role_complexity_limit}, "
                f"boolean={max_complexity}, count_numerical={max_complexity}, "
                f"distance_numerical={max_complexity}"
            )
            booleans, numericals, concepts, roles = dlplan_gen.generate_features(
                factory,
                list(self.states.values()),
                concept_complexity_limit,
                role_complexity_limit,
                max_complexity,
                max_complexity,
                max_complexity,
                3600,
                10000,
                **feature_generator_kwargs,
            )
            if config.get("extra_features", None):
                # Parsed directly with the factory rather than through generate_features above,
                # so these are exempt from concept_complexity_limit/role_complexity_limit/
                # max_complexity -- those limits are only enforced inside generate_features
                # itself, and appending after it returns is what achieves the exemption. They
                # keep DLPlan's own computed complexity (so the #minimize over feature cost
                # still prefers cheap elements over them where possible), and are appended
                # before pruning below, so compute_uninformative_*/compute_redundant_* still
                # see them exactly like a synthesised element.
                extra_booleans, extra_numericals, extra_concepts, extra_roles = _parse_extra_features(
                    factory, config["extra_features"]
                )
                added_booleans = _dedup_extend(booleans, extra_booleans)
                added_numericals = _dedup_extend(numericals, extra_numericals)
                added_concepts = _dedup_extend(concepts, extra_concepts)
                added_roles = _dedup_extend(roles, extra_roles)
                _log_added_extra_features(added_booleans, added_numericals, added_concepts, added_roles)
                self._extra_feature_keys.update(str(b) for b in added_booleans)
                self._extra_feature_keys.update(str(n) for n in added_numericals)
                self._extra_concept_keys.update(str(c) for c in added_concepts)
                self._extra_role_keys.update(str(r) for r in added_roles)
        self.features = {}
        self.concepts = {}
        self.roles = {}
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

    def node_to_action_augmented_state(self, problem: Problem, node: StateSpaceNode) -> None:
        self.node_id_to_action_aug_state_ids.setdefault((self.problem_name_to_id[problem.name], node.id), dict())
        for action in node.children.keys():
            fstate = get_action_augmented_state(problem, node.state, self.config, action)
            state_id = self.next_state_id
            self.next_state_id += 1
            self.states[fstate] = dlplan.core.State(
                state_id,
                self.instances[problem.name],
                [self.mappings[problem.name][fact] for fact in fstate],
            )
            self.node_id_to_action_aug_state_ids[(self.problem_name_to_id[problem.name], node.id)][action] = fstate
            self.state_id_to_node.setdefault(fstate, []).append(node)

    def node_to_param_augmented_state(self, problem: Problem, node: StateSpaceNode) -> None:
        self.node_id_to_param_aug_state_ids.setdefault((self.problem_name_to_id[problem.name], node.id), dict())
        for action in node.children.keys():
            self.node_id_to_param_aug_state_ids[(self.problem_name_to_id[problem.name], node.id)][action] = []
            for i, param in enumerate(action.parameters):
                fstate = get_param_augmented_state(problem, node.state, i, param)
                if fstate not in self.states:
                    state_id = self.next_state_id
                    self.next_state_id += 1
                    self.states[fstate] = dlplan.core.State(
                        state_id,
                        self.instances[problem.name],
                        [self.mappings[problem.name][fact] for fact in fstate],
                    )
                self.node_id_to_param_aug_state_ids[(self.problem_name_to_id[problem.name], node.id)][action].append(
                    fstate
                )
                self.state_id_to_node.setdefault(fstate, []).append(node)

    def node_to_state(self, problem: Problem, node: StateSpaceNode) -> None:
        state_id = self.next_state_id
        self.next_state_id += 1
        fstate = get_goal_augmented_state(problem, node.state)
        self.states[fstate] = dlplan.core.State(
            state_id,
            self.instances[problem.name],
            [self.mappings[problem.name][fact] for fact in fstate],
        )
        self.node_id_to_state_ids[(self.problem_name_to_id[problem.name], node.id)] = fstate
        self.state_id_to_node.setdefault(fstate, []).append(node)

    def generate_augmented_state_space(self, problem: Problem) -> StateSpaceGraph:
        return generate_state_space(self.domain, problem)

    def evaluate_concept(self, concept: str, problem: Problem, state: State) -> set[str]:
        return self.evaluate_concept_from_problem(concept, problem, state)

    def evaluate_role(self, role: str, problem: Problem, state: State) -> set[tuple[str, str]]:
        return self.evaluate_role_from_problem(role, problem, state)

    def evaluate_feature(self, feature: str, problem: Problem, state: State, action: Optional[Action] = None) -> bool:
        return self.evaluate_feature_from_problem(feature, problem, state, action)

    def get_augmented_dlplan_state(
        self, problem: Problem, state: State, action: Optional[Action] = None
    ) -> dlplan.core.State:
        return dlplan.core.State(
            -1,
            self.instances[problem.name],
            [
                self.mappings[problem.name][fact]
                for fact in get_action_augmented_state(problem, state, self.config, action)
            ],
        )

    def evaluate_feature_from_problem(
        self, feature: str, problem: Problem, state: State, action: Optional[Action] = None
    ) -> bool:
        feature = feature.strip('"')
        return self.features[feature].evaluate(self.get_augmented_dlplan_state(problem, state, action))

    def evaluate_role_from_problem(self, role: str, problem: Problem, state: State) -> set[tuple[str, str]]:
        role = role.strip('"')
        return set(
            [
                (self.obj_id_to_obj(problem, id1), self.obj_id_to_obj(problem, id2))
                for id1, id2 in self.roles[role]
                .evaluate(self.get_augmented_dlplan_state(problem, state))
                .to_sorted_vector()
            ]
        )

    def obj_id_to_obj(self, problem: Problem, obj_id: int) -> str:
        for obj in self.instances[problem.name].get_objects():
            if obj_id == obj.get_index():
                return obj.get_name()
        raise KeyError(f"Cannot find object with id {obj_id}")

    def evaluate_concept_from_problem(self, concept: str, problem: Problem, state: State) -> set[str]:
        concept = concept.strip('"')
        return set(
            [
                self.obj_id_to_obj(problem, id)
                for id in self.concepts[concept]
                .evaluate(self.get_augmented_dlplan_state(problem, state))
                .to_sorted_vector()
            ]
        )

    def is_concept_informative(self, concept_str: str) -> bool:
        for problem, state_graph in self.state_graphs.items():
            for node in state_graph.nodes.values():
                if node.goal and not self.config["include_goal_states"]:
                    continue
                if node.alive != Alive.ALIVE and not self.config["include_dead_states"]:
                    continue
                extension = self.evaluate_concept_from_problem(concept_str, self.problems[problem], node.state)
                if not extension:
                    return True
                for action in node.children.keys():
                    action_args = {str(p) for p in action.parameters}
                    if not action_args <= extension:
                        return True
        return False

    def compute_uninformative_concepts(self) -> set[str]:
        uninformative_concepts = set()
        for concept_str in self.concepts.keys():
            if not self.is_concept_informative(concept_str):
                uninformative_concepts.add(concept_str)
        log.info(f"Found {len(uninformative_concepts)} uninformative concept(s)")
        log.debug(", ".join(uninformative_concepts))
        return uninformative_concepts

    def is_role_informative(self, role_str: str) -> bool:
        for problem, state_graph in self.state_graphs.items():
            for node in state_graph.nodes.values():
                if node.goal and not self.config["include_goal_states"]:
                    continue
                if node.alive != Alive.ALIVE and not self.config["include_dead_states"]:
                    continue
                extension = self.evaluate_role_from_problem(role_str, self.problems[problem], node.state)
                if not extension:
                    return True
                log.debug(f"Role {role_str} extension: {extension}")
                for action in node.children.keys():
                    action_args = {str(p) for p in action.parameters}
                    for combination in itertools.permutations(action_args, 2):
                        log.debug(f"Checking action argument combination {combination}")
                        if combination not in extension:
                            log.debug(f"Action argument combination {combination} not in extension")
                            return True
        return False

    def compute_uninformative_roles(self) -> set[str]:
        uninformative_roles = set()
        for role_str in self.roles.keys():
            if not self.is_role_informative(role_str):
                uninformative_roles.add(role_str)
        log.info(f"Found {len(uninformative_roles)} uninformative role(s)")
        log.debug(", ".join(uninformative_roles))
        return uninformative_roles

    def is_feature_informative(self, feature: Feature) -> bool:
        has_false = False
        has_true = False
        for fstate, state in self.states.items():
            if all(
                [
                    node.goal
                    and not self.config["include_goal_states"]
                    or node.alive != Alive.ALIVE
                    and not self.config["include_dead_states"]
                    for node in self.state_id_to_node[fstate]
                ]
            ):
                continue
            eval = feature.evaluate(state)
            if eval:
                has_true = True
            else:
                has_false = True
            if has_true and has_false:
                return True
        return has_true and has_false

    def compute_redundant_features(self) -> set[str]:
        redundant_features = set()
        feature_evals = set()
        for feature_str, feature in self.features.items():
            true_states = frozenset([state for state in self.states.values() if feature.evaluate(state)])
            if true_states in feature_evals:
                redundant_features.add(feature_str)
            else:
                feature_evals.add(true_states)
        log.info(f"Found {len(redundant_features)} redundant feature(s)")
        log.debug(", ".join(redundant_features))
        return redundant_features

    def compute_redundant_concepts(self) -> set[str]:
        evals: MutableMapping[str, Mapping[Tuple[str, State], Collection]] = dict()
        redundant_concepts = set()
        for concept_str in self.concepts.keys():
            eval: MutableMapping[Tuple[str, State], Collection[str]] = dict()
            for problem, state_graph in self.state_graphs.items():
                for node in state_graph.nodes.values():
                    # eval[(problem, node.state)] = frozenset()
                    eval[(problem, node.state)] = frozenset(
                        self.evaluate_concept_from_problem(concept_str, self.problems[problem], node.state)
                    )
            if eval in evals.values():
                redundant_concepts.add(concept_str)
            else:
                evals[concept_str] = eval
        log.info(f"Found {len(redundant_concepts)} redundant concept(s)")
        log.debug(", ".join(redundant_concepts))
        return redundant_concepts

    def compute_redundant_roles(self) -> set[str]:
        evals: MutableMapping[str, Mapping[Tuple[str, State], Collection]] = dict()
        redundant_roles = set()
        for role_str in self.roles.keys():
            eval: MutableMapping[Tuple[str, State], Collection[tuple[str, str]]] = dict()
            for problem, state_graph in self.state_graphs.items():
                for node in state_graph.nodes.values():
                    eval[(problem, node.state)] = frozenset(
                        self.evaluate_role_from_problem(role_str, self.problems[problem], node.state)
                    )
            if eval in evals.values():
                redundant_roles.add(role_str)
            else:
                evals[role_str] = eval
        log.info(f"Found {len(redundant_roles)} redundant role(s)")
        log.debug(", ".join(redundant_roles))
        return redundant_roles

    def compute_uninformative_features(self) -> set[str]:
        uninformative_features = set()
        for feature_str, feature in self.features.items():
            if not self.is_feature_informative(feature):
                uninformative_features.add(feature_str)
        log.info(f"Found {len(uninformative_features)} uninformative feature(s)")
        log.debug(", ".join(uninformative_features))
        return uninformative_features

    def is_concept_static(self, concept_str: str) -> bool:
        extensions = set()
        for problem, state_graph in self.state_graphs.items():
            for node in state_graph.nodes.values():
                if node.goal and not self.config["include_goal_states"]:
                    continue
                if node.alive != Alive.ALIVE and not self.config["include_dead_states"]:
                    continue
                extensions.add(
                    frozenset(self.evaluate_concept_from_problem(concept_str, self.problems[problem], node.state))
                )
                if len(extensions) > 1:
                    return True
        return False

    def compute_static_concepts(self) -> set[str]:
        static_concepts = set()
        for concept_str in self.concepts.keys():
            if self.is_concept_static(concept_str):
                static_concepts.add(concept_str)
        log.info(f"Found {len(static_concepts)} static concept(s)")
        log.debug(", ".join(static_concepts))
        return static_concepts

    def is_role_static(self, role_str: str) -> bool:
        extensions = set()
        for problem, state_graph in self.state_graphs.items():
            for node in state_graph.nodes.values():
                if node.goal and not self.config["include_goal_states"]:
                    continue
                if node.alive != Alive.ALIVE and not self.config["include_dead_states"]:
                    continue
                extensions.add(
                    frozenset(self.evaluate_role_from_problem(role_str, self.problems[problem], node.state))
                )
                if len(extensions) > 1:
                    return True
        return False

    def compute_static_roles(self) -> set[str]:
        static_roles = set()
        for role_str in self.roles.keys():
            if self.is_role_static(role_str):
                static_roles.add(role_str)
        log.info(f"Found {len(static_roles)} static role(s)")
        log.debug(", ".join(static_roles))
        return static_roles

    def lookup_node(self, problem_id: int, node_id: int) -> tuple[Problem, StateSpaceNode]:
        """Resolve an ASP (instance, state) pair back to its problem and state-space node.

        `node.id` values have gaps after `StateSpaceGraph.prune_nodes`, so this indexes the
        nodes rather than assuming a dense numbering.
        """
        problem_name = self.problem_id_to_name[problem_id]
        state_graph = self.state_graphs[problem_name]
        index = self._node_indices.get(problem_name)
        if index is None:
            index = {node.id: node for node in state_graph.nodes.values()}
            self._node_indices[problem_name] = index
        return self.problems[problem_name], index[node_id]

    def node_to_clingo(self, problem: Problem, node: StateSpaceNode, stats: dict) -> str:
        problem_id = self.problem_name_to_id[problem.name]
        clingo_program = ""
        clingo_program += (
            f"% " + ",".join([f'{p.name}({",".join([str(p) for p in p.terms])})' for p in node.state]) + "\n"
        )
        clingo_program += f"state({problem_id}, {node.id}).\n"
        if node.alive == Alive.PRUNED:
            clingo_program += f"pruned({problem_id}, {node.id}).\n"
            return clingo_program
        if node.alive == Alive.ALIVE:
            clingo_program += f"alive({problem_id}, {node.id}).\n"
        if node.alive != Alive.ALIVE and not self.config["include_dead_states"]:
            return clingo_program
        is_goal = check_formula(node.state, problem.goal)
        if is_goal:
            clingo_program += f"goal({problem_id}, {node.id}).\n"
            if not self.config["include_goal_states"]:
                return clingo_program
        # Exactly the states the good_trans choice rule and the good_sig/bad_sig rules range
        # over (`alive(I,S), not goal(I,S)`); see forced_labels.
        selectable_state = node.alive == Alive.ALIVE and not is_goal
        if self.config["include_actions"]:
            for action, aug_state in self.node_id_to_action_aug_state_ids[(problem_id, node.id)].items():
                aug_state_id = self.states[aug_state].get_index()
                action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
                clingo_program += f"aug_state({problem_id}, {node.id}, {action_str}, {aug_state_id}).\n"
                for feature_str, feature in self.features.items():
                    if feature_str in stats["uninformative_features"] or feature_str in stats["redundant_features"]:
                        stats["num_skipped_feature_evals"] += 1
                        continue
                    feature_str = f'"{feature_str}"'
                    eval = feature.evaluate(self.states[aug_state])
                    if type(eval) is bool:
                        eval = 1 if eval else 0
                    clingo_program += f"eval({aug_state_id}, {feature_str}, {eval}).\n"
                    stats["num_feature_evals"] += 1
        sig_enabled = bool(self.config.get("emit_action_signatures", False))
        # c_eval/4, r_eval/5, aparam/3 and aname/2 are read back by solve_datalog.lp and the
        # datalog-actions/datalog-action-params variants, but solve_datalog_sig.lp never
        # references any of the four -- the separation layer is reconstructed in Python from
        # the same dlplan evaluations below (sig_concepts/sig_roles/sig_bools), independent of
        # whether the text is emitted. See emit_object_facts in config/default.yaml.
        emit_object_facts = bool(self.config.get("emit_object_facts", True))
        sig_bools: list[tuple[str, int]] = []
        if self.config["include_pristine_states"]:
            for feature_str, feature in self.features.items():
                if feature_str in stats["uninformative_features"] or feature_str in stats["redundant_features"]:
                    stats["num_skipped_feature_evals"] += 1
                    continue
                raw_feature_str = feature_str
                feature_str = f'"{feature_str}"'
                eval = feature.evaluate(self.states[self.node_id_to_state_ids[(problem_id, node.id)]])
                if type(eval) is bool:
                    eval = 1 if eval else 0
                if sig_enabled:
                    # Mirrors bool_eval/4 in the solve program, which is eval/4 thresholded at 0.
                    sig_bools.append((raw_feature_str, 1 if eval > 0 else 0))
                clingo_program += f"eval({problem_id}, {node.id}, {feature_str}, {eval}).\n"
                stats["num_feature_evals"] += 1
        if self.config["include_action_params"]:
            for action, aug_states in self.node_id_to_param_aug_state_ids[(problem_id, node.id)].items():
                action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
                for i, aug_state in enumerate(aug_states):
                    aug_state_id = self.states[aug_state].get_index()
                    clingo_program += f"aug_state({problem_id}, {node.id}, {action_str}, {i}, {aug_state_id}).\n"
                    for feature_str, feature in self.features.items():
                        if (
                            feature_str in stats["uninformative_features"]
                            or feature_str in stats["redundant_features"]
                        ):
                            stats["num_skipped_feature_evals"] += 1
                            continue
                        if not get_aparam_predicate_name(0) in feature_str:
                            continue
                        feature_str = f'"{feature_str}"'
                        eval = feature.evaluate(self.states[aug_state])
                        if type(eval) is bool:
                            eval = 1 if eval else 0
                        clingo_program += f"aug_eval({aug_state_id}, {feature_str}, {eval}).\n"
                        stats["num_feature_evals"] += 1
        all_action_args = {str(p) for action in node.children.keys() for p in action.parameters}
        sig_params = {action: [str(p) for p in action.parameters] for action in node.children}
        sig_concepts: dict[Action, set[tuple[str, int]]] = {action: set() for action in node.children}
        sig_roles: dict[Action, set[tuple[str, int, int]]] = {action: set() for action in node.children}
        for concept_str, concept in self.concepts.items():
            if concept_str in stats["uninformative_concepts"]:
                # log.debug(f'Concept {concept_str} does not distinguish any action arguments, skipping')
                stats["num_skipped_concept_evals"] += len(
                    self.evaluate_concept_from_problem(f'"{concept_str}"', problem, node.state)
                )
                continue
            if concept_str in stats["static_concepts"]:
                # log.debug(f'Concept {concept_str} is static, skipping')
                stats["num_skipped_concept_evals"] += len(
                    self.evaluate_concept_from_problem(f'"{concept_str}"', problem, node.state)
                )
                continue
            if concept_str in stats["redundant_concepts"]:
                # log.debug(f'Concept {concept_str} is redundant, skipping')
                stats["num_skipped_concept_evals"] += len(
                    self.evaluate_concept_from_problem(f'"{concept_str}"', problem, node.state)
                )
                continue
            raw_concept_str = concept_str
            concept_str = f'"{concept_str}"'
            extension = self.evaluate_concept_from_problem(concept_str, problem, node.state)
            if sig_enabled:
                for action, params in sig_params.items():
                    for index, param in enumerate(params):
                        if param in extension:
                            sig_concepts[action].add((raw_concept_str, index))
            for obj in extension:
                if obj in all_action_args:
                    if emit_object_facts:
                        clingo_program += f'c_eval({problem_id}, {node.id}, {concept_str}, "{obj}").\n'
                    stats["num_concept_evals"] += 1
                else:
                    stats["num_skipped_concept_evals"] += 1
        for role_str, role in self.roles.items():
            if role_str in stats["uninformative_roles"]:
                # log.debug(f'Role {role_str} does not distinguish any action argument pairs, skipping')
                stats["num_skipped_role_evals"] += len(
                    self.evaluate_role_from_problem(f'"{role_str}"', problem, node.state)
                )
                continue
            if role_str in stats["static_roles"]:
                # log.debug(f'Role {role_str} is static, skipping')
                stats["num_skipped_role_evals"] += len(
                    self.evaluate_role_from_problem(f'"{role_str}"', problem, node.state)
                )
                continue
            if role_str in stats["redundant_roles"]:
                # log.debug(f'Role {role_str} is redundant, skipping')
                stats["num_skipped_role_evals"] += len(
                    self.evaluate_role_from_problem(f'"{role_str}"', problem, node.state)
                )
                continue
            raw_role_str = role_str
            role_str = f'"{role_str}"'
            role_extension = self.evaluate_role_from_problem(role_str, problem, node.state)
            if sig_enabled:
                for action, params in sig_params.items():
                    for index1, param1 in enumerate(params):
                        for index2, param2 in enumerate(params):
                            if (param1, param2) in role_extension:
                                sig_roles[action].add((raw_role_str, index1, index2))
            for obj1, obj2 in role_extension:
                if obj1 in all_action_args and obj2 in all_action_args:
                    if emit_object_facts:
                        clingo_program += f'r_eval({problem_id}, {node.id}, {role_str}, "{obj1}", "{obj2}").\n'
                    stats["num_role_evals"] += 1
                else:
                    stats["num_skipped_role_evals"] += 1
        for action, children in node.children.items():
            action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
            if emit_object_facts:
                clingo_program += f'aname({action_str}, "{action.name}").\n'
            if sig_enabled:
                signature = ActionSignature(
                    action.name,
                    len(action.parameters),
                    frozenset(sig_concepts[action]),
                    frozenset(sig_roles[action]),
                    tuple(sig_bools),
                )
                signature_id = self.signature_ids.setdefault(signature, len(self.signature_ids))
                clingo_program += f"asig({problem_id}, {node.id}, {action_str}, {signature_id}).\n"
                if selectable_state:
                    self._occurrences.append(
                        ActionOccurrence(
                            problem_id,
                            node.id,
                            action_str,
                            signature_id,
                            # good_trans is unselectable for an action with an outcome that is
                            # neither alive nor pruned; those are the only two node kinds that
                            # emit alive/2 or pruned/2 above.
                            all(child.alive in (Alive.ALIVE, Alive.PRUNED) for child in children),
                        )
                    )
            if selectable_state and self.config.get("plan_label_heuristic", False):
                if any(suffix and suffix[0] == action for suffix in node.plan_suffixes):
                    self._plan_actions.append((problem_id, node.id, action_str))
            for child in children:
                clingo_program += f"trans({problem_id}, {node.id}, {action_str}, {child.id}).\n"
                if emit_object_facts and (self.concepts or self.roles):
                    params = [f'"{p}"' for p in action.parameters]
                    for i, p in enumerate(params):
                        clingo_program += f"aparam({action_str}, {i}, {p}).\n"
        return clingo_program

    @property
    def signatures(self) -> list[ActionSignature]:
        """The signature classes collected while writing the per-node facts, indexed by id."""
        return [signature for signature, _ in sorted(self.signature_ids.items(), key=lambda item: item[1])]

    def _emit_action_signatures(self) -> str:
        """Emit the separation layer over the signature classes.

        The datalog separation constraint observes an (instance, state, action) triple only
        through the action name, the concept membership of each argument position, the role
        membership of each ordered position pair, and the source state's boolean feature
        vector. Triples sharing all of that can never be told apart by any selection, so the
        constraint may range over these classes instead of over the triples themselves --
        quadratically fewer. `asig/4` links the two layers.

        Which elements distinguish a given pair of classes is fixed by the instance, so it is
        computed in Python (see `action_signatures`) and emitted as `sig_pair/3` plus a
        deduplicated `dist/2` relation, instead of being derived by rules that would pair every
        class pair with every feature, concept and role.

        With `lazy_pairs` that relation is left out entirely and handed to the solver one batch
        of violated pairs at a time instead (see `lazy_pairs.py`); only `asig/4` is emitted here.
        """
        if self.config.get("lazy_pairs", False):
            return ""
        return "".join(iter_dist_set_facts(self.signatures))

    def _emit_forced_labels(self) -> str:
        """Emit ``forced_good/3`` and ``forced_bad/3`` for the labels fixed by the graph layer.

        See `forced_labels`: these are the (instance, state, action) occurrences whose good/bad
        label is the same in every model of the program, so pinning them removes choice points
        without removing models. `solve_datalog_sig.lp` drops the forced-bad ones from the
        ``good_trans`` choice and adds ``:- forced_good(I,S,A), not good_action(I,S,A).``

        Both predicates are ``#defined`` there, so with the flag off nothing is emitted and the
        ground program is exactly the one produced before.
        """
        forced = compute_forced_labels(self._occurrences)
        self.forced_labels = forced
        log.info(f"Forced labels: {forced.summary()}")
        if forced.inconsistent:
            log.warning(
                "The forced-label fixpoint is contradictory: this instance has no policy at all."
                " The facts are emitted anyway, so clingo reports the refutation itself."
            )
        return "".join(
            [
                f"forced_good({occurrence.instance}, {occurrence.state}, {occurrence.action}).\n"
                for occurrence in sorted(forced.good)
            ]
            + [
                f"forced_bad({occurrence.instance}, {occurrence.state}, {occurrence.action}).\n"
                for occurrence in sorted(forced.bad)
            ]
        )

    def _emit_plan_actions(self) -> str:
        """Emit ``plan_action/3`` for the actions that start an example-plan suffix at a state.

        Consumed only by the ``plan_heuristic`` part of `solve_datalog_sig.lp`, a ``#heuristic``
        directive that biases the ``good_trans`` decision towards the plan. A heuristic changes
        the order of the search, never its solution space, so the optimum is unaffected.
        """
        log.info(f"Plan-label heuristic: {len(self._plan_actions)} on-plan action(s)")
        return "".join(
            f"plan_action({instance}, {state}, {action}).\n" for instance, state, action in self._plan_actions
        )

    def to_clingo(self) -> str:
        stats = {
            "num_skipped_feature_evals": 0,
            "num_feature_evals": 0,
            "num_concept_evals": 0,
            "num_skipped_concept_evals": 0,
            "num_role_evals": 0,
            "num_skipped_role_evals": 0,
            "uninformative_features": (
                self.compute_uninformative_features() if self.config["prune_features"] else set()
            ),
            "uninformative_concepts": (
                self.compute_uninformative_concepts() if self.config["prune_concepts"] else set()
            ),
            "uninformative_roles": (self.compute_uninformative_roles() if self.config["prune_roles"] else set()),
            "static_concepts": (self.compute_static_concepts() if self.config["prune_static_concepts"] else set()),
            "static_roles": (self.compute_static_roles() if self.config["prune_static_roles"] else set()),
            "redundant_features": (
                self.compute_redundant_features() if self.config["prune_redundant_features"] else set()
            ),
            "redundant_concepts": (
                self.compute_redundant_concepts() if self.config["prune_redundant_concepts"] else set()
            ),
            "redundant_roles": (self.compute_redundant_roles() if self.config["prune_redundant_roles"] else set()),
        }
        extra_complexity = self.config.get("extra_features_complexity")
        if extra_complexity is not None and (
            self._extra_feature_keys or self._extra_concept_keys or self._extra_role_keys
        ):
            log.info(
                f"extra_features_complexity={extra_complexity}: overriding the emitted complexity "
                f"for extra_features element(s) {sorted(self._extra_feature_keys | self._extra_concept_keys | self._extra_role_keys)}"
            )
        clingo_program = ""
        for feature_str, feature in self.features.items():
            complexity = (
                extra_complexity
                if extra_complexity is not None and feature_str in self._extra_feature_keys
                else feature.compute_complexity()
            )
            feature_str = f'"{feature_str}"'
            clingo_program += f"feature({feature_str}).\n"
            clingo_program += f"feature_complexity({feature_str}, {complexity}).\n"
        for concept_str, concept in self.concepts.items():
            complexity = (
                extra_complexity
                if extra_complexity is not None and concept_str in self._extra_concept_keys
                else concept.compute_complexity()
            )
            concept_str = f'"{concept_str}"'
            clingo_program += f"concept({concept_str}).\n"
            clingo_program += f"concept_complexity({concept_str}, {complexity}).\n"
        for role_str, role in self.roles.items():
            complexity = (
                extra_complexity
                if extra_complexity is not None and role_str in self._extra_role_keys
                else role.compute_complexity()
            )
            role_str = f'"{role_str}"'
            clingo_program += f"role({role_str}).\n"
            clingo_program += f"role_complexity({role_str}, {complexity}).\n"
        for state_graph in self.state_graphs.values():
            for node in state_graph.nodes.values():
                clingo_program += self.node_to_clingo(state_graph.problem, node, stats)
        if self.config.get("emit_action_signatures", False):
            clingo_program += self._emit_action_signatures()
            log.info(f"Collapsed the separation layer to {len(self.signature_ids)} action signature(s)")
            if self.config.get("fix_forced_labels", False):
                clingo_program += self._emit_forced_labels()
        if self.config.get("plan_label_heuristic", False):
            clingo_program += self._emit_plan_actions()
        log.info(
            f'Generated program with {stats["num_feature_evals"]} feature evaluations ({stats["num_skipped_feature_evals"]} skipped), '
            f'{stats["num_concept_evals"]} concept evaluations ({stats["num_skipped_concept_evals"]} skipped), '
            f'and {stats["num_role_evals"]} role evaluations ({stats["num_skipped_role_evals"]} skipped)'
        )
        return clingo_program
