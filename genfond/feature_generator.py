from dlplan.core import VocabularyInfo, InstanceInfo, State, \
    SyntacticElementFactory
import dlplan.generator as dlplan_gen
from pddl.logic import Predicate
from .ground import ground_domain_predicates, ground
from .state_space_generator import apply_effects, generate_state_space, check_formula, Alive
import logging
import itertools

log = logging.getLogger(__name__)

MAX_ACTION_PARAMETERS = 4


def get_predicate_name_from_action(action):
    return f'A_{action.name}'


def get_predicate_from_action(action):
    return Predicate(get_predicate_name_from_action(action), *action.parameters)


def construct_vocabulary_info(domain, include_actions=False):
    vocabulary = VocabularyInfo()
    for predicate in domain.predicates:
        assert f'{predicate.name}_G' not in [p.name for p in domain.predicates]
    for predicate in domain.predicates:
        # TODO some predicates may be static.
        vocabulary.add_predicate(predicate.name, predicate.arity)
        vocabulary.add_predicate(f'{predicate.name}_G', predicate.arity)
    if include_actions:
        for action in domain.actions:
            action_predicate = get_predicate_name_from_action(action)
            assert action_predicate not in [p.name for p in domain.predicates]
            vocabulary.add_predicate(action_predicate, len(action.parameters))
    for constant in domain.constants:
        vocabulary.add_constant(constant.name)
    return vocabulary


def _get_state_from_goal(goal_formula):
    states = apply_effects(set([frozenset()]), goal_formula)
    assert len(states) == 1, \
        f'Goal formula must define a unique goal state, found {len(states)} states: {states}'
    state = next(iter(states))
    goal_state = {Predicate(f'{predicate.name}_G', *predicate.terms) for predicate in state}
    return goal_state


def construct_instance_info(vocabulary, domain, problem, problem_id, include_actions=False):
    instance = InstanceInfo(problem_id, vocabulary)
    map = dict()
    for object in problem.objects:
        instance.add_object(object.name)
    for predicate in ground_domain_predicates(domain, problem):
        map[predicate] = instance.add_atom(predicate.name, [str(t) for t in predicate.terms])
        goal_predicate = Predicate(f'{predicate.name}_G', *predicate.terms)
        map[goal_predicate] = instance.add_atom(goal_predicate.name, [str(t) for t in predicate.terms])
    if include_actions:
        for action in ground(domain, problem):
            map[get_predicate_from_action(action)] = instance.add_atom(get_predicate_name_from_action(action),
                                                                       [str(t) for t in action.parameters])
    return instance, map


def get_augmented_state(domain, problem, state, config):
    augmented_state = set(state | _get_state_from_goal(problem.goal))
    if config['include_actions']:
        for action in ground(domain, problem):
            if check_formula(state, action.precondition):
                augmented_state.add(get_predicate_from_action(action))
    return frozenset(augmented_state)


class FeaturePool:

    def __init__(
        self,
        domain,
        problems,
        config,
        max_complexity=None,
        all_generators=False,
    ):
        assert len({problem.name for problem in problems}) == len(problems), \
            "Problem names must be unique."
        self.domain = domain
        self.problems = {problem.name: problem for problem in problems}
        self.config = config
        self.problem_name_to_id = {problem.name: i for i, problem in enumerate(problems)}
        vocabulary = construct_vocabulary_info(domain, include_actions=config['include_actions'])
        self.states = dict()
        self.state_graphs = dict()
        self.goal_states = dict()
        self.instances = dict()
        if not max_complexity:
            max_complexity = config['max_complexity']
        for problem in problems:
            instance, mapping = construct_instance_info(vocabulary,
                                                        domain,
                                                        problem,
                                                        self.problem_name_to_id[problem.name],
                                                        include_actions=config['include_actions'])
            self.state_graphs[problem.name] = generate_state_space(domain, problem)
            self.goal_states[problem.name] = _get_state_from_goal(problem.goal)
            self.instances[problem.name] = instance
            pddl_states = set()
            ground_actions = ground(domain, problem)
            for node in self.state_graphs[problem.name].nodes.values():
                pddl_states.add(get_augmented_state(domain, problem, node.state, config))
            self.states[problem.name] = {
                state: State(-1, instance, [mapping[predicate] for predicate in state])
                for state in pddl_states
            }
        factory = SyntacticElementFactory(vocabulary)
        if config.get('preset_features', None):
            str_gens = config['preset_features']
        else:
            if all_generators:
                feature_generator_kwargs = config['unrestricted_feature_generator']
            else:
                feature_generator_kwargs = config['feature_generator']
            booleans, numericals, concepts, roles = dlplan_gen.generate_features(
                factory, [state for state in self.states[problem.name].values() for problem in problems],
                *5 * [max_complexity], 3600, 10000, **feature_generator_kwargs)
        self.features = {}
        self.concepts = {}
        self.roles = {}
        if config['include_boolean_features']:
            for feature in booleans:
                self.features[str(feature)] = feature
        if config['include_numerical_features']:
            for feature in numericals:
                self.features[str(feature)] = feature
        if config['include_concepts']:
            self.concepts = {str(concept): concept for concept in concepts}
        if config['include_roles']:
            self.roles = {str(role): role for role in roles}
        log.debug(f'generated concepts: {", ".join(self.concepts.keys())}')
        log.debug(f'generated roles: {", ".join(self.roles.keys())}')
        log.debug(f'generated features: {", ".join(self.features.keys())}')

    def evaluate_feature(self, feature, state):
        # Try to guess which problem the state belongs to.
        problems = [
            problem for (problem, states) in self.states.items()
            if get_augmented_state(self.domain, self.problems[problem], state, self.config) in states
        ]
        assert len(problems) == 1, \
            'Cannot determine which problem the state belongs to,' \
            f' found {len(problems)} matching problems'
        return self.evaluate_feature_from_problem(feature, problems[0], state)

    def evaluate_concept(self, concept, state):
        # Try to guess which problem the state belongs to.
        problems = [
            problem for (problem, states) in self.states.items()
            if get_augmented_state(self.domain, self.problems[problem], state, self.config) in states
        ]
        assert len(problems) == 1, \
            'Cannot determine which problem the state belongs to,' \
            f' found {len(problems)} matching problems'
        return self.evaluate_concept_from_problem(concept, problems[0], state)

    def evaluate_role(self, role, state):
        # Try to guess which problem the state belongs to.
        problems = [
            problem for (problem, states) in self.states.items()
            if get_augmented_state(self.domain, self.problems[problem], state, self.config) in states
        ]
        assert len(problems) == 1, \
            'Cannot determine which problem the state belongs to,' \
            f' found {len(problems)} matching problems'
        return self.evaluate_role_from_problem(role, problems[0], state)

    def evaluate_feature_from_problem(self, feature, problem, state):
        feature = feature.strip('"')
        return self.features[feature].evaluate(self.states[problem][get_augmented_state(
            self.domain, self.problems[problem], state, self.config)])

    def evaluate_role_from_problem(self, role, problem, state):
        role = role.strip('"')
        return set([(self.obj_id_to_obj(problem, id1), self.obj_id_to_obj(problem, id2))
                    for id1, id2 in self.roles[role].evaluate(self.states[problem][get_augmented_state(
                        self.domain, self.problems[problem], state, self.config)]).to_sorted_vector()])

    def obj_id_to_obj(self, problem, obj_id):
        for obj in self.instances[problem].get_objects():
            if obj_id == obj.get_index():
                return obj.get_name()
        return None

    def evaluate_concept_from_problem(self, concept, problem, state):
        concept = concept.strip('"')
        return set([
            self.obj_id_to_obj(problem, id)
            for id in self.concepts[concept].evaluate(self.states[problem][get_augmented_state(
                self.domain, self.problems[problem], state, self.config)]).to_sorted_vector()
        ])

    def is_concept_informative(self, concept_str):
        for problem, state_graph in self.state_graphs.items():
            for node in state_graph.nodes.values():
                extension = self.evaluate_concept_from_problem(concept_str, problem, node.state)
                if not extension:
                    return True
                for action in node.children.keys():
                    action_args = {str(p) for p in action.parameters}
                    if not action_args <= extension:
                        return True
        return False

    def compute_uninformative_concepts(self):
        uninformative_concepts = set()
        for concept_str in self.concepts.keys():
            if not self.is_concept_informative(concept_str):
                uninformative_concepts.add(concept_str)
        log.info(f'Found {len(uninformative_concepts)} uninformative concept(s): {", ".join(uninformative_concepts)}')
        return uninformative_concepts

    def is_role_informative(self, role_str):
        for problem, state_graph in self.state_graphs.items():
            for node in state_graph.nodes.values():
                extension = self.evaluate_role_from_problem(role_str, problem, node.state)
                if not extension:
                    return True
                log.debug(f'Role {role_str} extension: {extension}')
                for action in node.children.keys():
                    action_args = {str(p) for p in action.parameters}
                    for combination in itertools.permutations(action_args, 2):
                        log.debug(f'Checking action argument combination {combination}')
                        if combination not in extension:
                            log.debug(f'Action argument combination {combination} not in extension')
                            return True
        return False

    def compute_uninformative_roles(self):
        uninformative_roles = set()
        for role_str in self.roles.keys():
            if not self.is_role_informative(role_str):
                uninformative_roles.add(role_str)
        log.info(f'Found {len(uninformative_roles)} uninformative role(s): {", ".join(uninformative_roles)}')
        return uninformative_roles

    def to_clingo(self):
        clingo_program = ""
        for feature_str, feature in self.features.items():
            feature_str = f'"{feature_str}"'
            clingo_program += f'feature({feature_str}).\n'
            clingo_program += f'feature_complexity({feature_str}, {feature.compute_complexity()}).\n'
        for concept_str, concept in self.concepts.items():
            concept_str = f'"{concept_str}"'
            clingo_program += f'concept({concept_str}).\n'
            clingo_program += f'concept_complexity({concept_str}, {concept.compute_complexity()}).\n'
        for role_str, role in self.roles.items():
            role_str = f'"{role_str}"'
            clingo_program += f'role({role_str}).\n'
            clingo_program += f'role_complexity({role_str}, {role.compute_complexity()}).\n'
        num_feature_evals = 0
        num_concept_evals = 0
        num_skipped_concept_evals = 0
        num_role_evals = 0
        num_skipped_role_evals = 0
        uninformative_concepts = self.compute_uninformative_concepts() if self.config['prune_concepts'] else set()
        uninformative_roles = self.compute_uninformative_roles() if self.config['prune_roles'] else set()
        for problem, state_graph in self.state_graphs.items():
            problem_id = self.problem_name_to_id[problem]
            for node in state_graph.nodes.values():
                clingo_program += f'state({problem_id}, {node.id}).\n'
                if node.alive == Alive.ALIVE:
                    clingo_program += f'alive({problem_id}, {node.id}).\n'
                if node.alive != Alive.ALIVE and not self.config['include_dead_states']:
                    continue
                if check_formula(node.state, state_graph.problem.goal):
                    clingo_program += f'goal({problem_id}, {node.id}).\n'
                for feature_str, feature in self.features.items():
                    feature_str = f'"{feature_str}"'
                    eval = self.evaluate_feature_from_problem(feature_str, problem, node.state)
                    if type(eval) is bool:
                        eval = 1 if eval else 0
                    clingo_program += f'eval({problem_id}, {node.id}, {feature_str}, {eval}).\n'
                    num_feature_evals += 1
                all_action_args = {str(p) for action in node.children.keys() for p in action.parameters}
                for concept_str, concept in self.concepts.items():
                    if concept_str in uninformative_concepts:
                        #log.debug(f'Concept {concept_str} does not distinguish any action arguments, skipping')
                        num_skipped_concept_evals += 1
                        continue
                    concept_str = f'"{concept_str}"'
                    extension = self.evaluate_concept_from_problem(concept_str, problem, node.state)
                    for obj in extension:
                        clingo_program += f'c_eval({problem_id}, {node.id}, {concept_str}, "{obj}").\n'
                        num_concept_evals += 1
                for role_str, role in self.roles.items():
                    if role_str in uninformative_roles:
                        #log.debug(f'Role {role_str} does not distinguish any action argument pairs, skipping')
                        num_skipped_role_evals += 1
                        continue
                    role_str = f'"{role_str}"'
                    for obj1, obj2 in self.evaluate_role_from_problem(role_str, problem, node.state):
                        clingo_program += f'r_eval({problem_id}, {node.id}, {role_str}, "{obj1}", "{obj2}").\n'
                        num_role_evals += 1
                for action, children in node.children.items():
                    action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
                    for child in children:
                        clingo_program += f'trans({problem_id}, {node.id}, {action_str}, {child.id}).\n'
                        if self.concepts:
                            params = [f'"{p}"' for p in action.parameters]
                            assert len(params) <= MAX_ACTION_PARAMETERS, \
                                    f'Action {action.name} has too many parameters: {len(params)}'
                            clingo_program += f'amap({action_str}, "{action.name}", {", ".join(params)}).\n'
        log.info(f'Generated program with {num_feature_evals} feature evaluations, '
                 f'{num_concept_evals} concept evaluations ({num_skipped_concept_evals} skipped), '
                 f'and {num_role_evals} role evaluations ({num_skipped_role_evals} skipped)')
        return clingo_program
