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


def get_aparam_predicate_name(i):
    return f'aparam{i}'


def get_aparam_predicate(action, i):
    return Predicate(get_aparam_predicate_name(i), action.parameters[i])


def construct_vocabulary_info(domain, include_actions=False):
    vocabulary = VocabularyInfo()
    for predicate in domain.predicates:
        assert f'{predicate.name}_G' not in [p.name for p in domain.predicates]
    for predicate in domain.predicates:
        # TODO some predicates may be static.
        vocabulary.add_predicate(predicate.name, predicate.arity)
        vocabulary.add_predicate(f'{predicate.name}_G', predicate.arity)
    max_arity = max([len(action.parameters) for action in domain.actions])
    if include_actions:
        for i in range(max_arity):
            log.debug(f'Adding action parameter predicate {get_aparam_predicate_name(i)}')
            aparam_pred = get_aparam_predicate_name(i)
            assert aparam_pred not in [p.name for p in domain.predicates]
            vocabulary.add_predicate(aparam_pred, 1)
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
            for i, param in enumerate(action.parameters):
                map[get_aparam_predicate(action, i)] = instance.add_atom(get_aparam_predicate_name(i), [str(param)])
    return instance, map


def get_goal_augmented_state(problem, state):
    return frozenset(state | _get_state_from_goal(problem.goal))


def get_augmented_state(problem, state, config, action=None):
    assert not config['include_actions'] or action, 'Action must be provided when including actions'
    augmented_state = get_goal_augmented_state(problem, state)
    if config['include_actions']:
        param_atoms = {get_aparam_predicate(action, i) for i, _ in enumerate(action.parameters)}
        augmented_state |= param_atoms
    action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"' if action else 'None'
    log.debug(f'Generated augmented state (action={action_str}): [{", ".join([str(p) for p in augmented_state])}]')
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
        log.debug(f'Constructed vocabulary: {vocabulary}')
        self.states = dict()
        self.node_id_to_state_ids = dict()
        self.state_graphs = dict()
        self.instances = dict()
        self.mappings = dict()
        next_state_id = 0
        if not max_complexity:
            max_complexity = config['max_complexity']
        for problem in problems:
            instance, mapping = construct_instance_info(vocabulary,
                                                        domain,
                                                        problem,
                                                        self.problem_name_to_id[problem.name],
                                                        include_actions=config['include_actions'])
            self.state_graphs[problem.name] = generate_state_space(domain, problem)
            self.instances[problem.name] = instance
            self.mappings[problem.name] = mapping
            new_states = set()
            ground_actions = ground(domain, problem)
            for node in self.state_graphs[problem.name].nodes.values():
                if config['include_actions']:
                    self.node_id_to_state_ids[(self.problem_name_to_id[problem.name], node.id)] = dict()
                    for action in node.children.keys():
                        state_id = next_state_id
                        next_state_id += 1
                        self.states[state_id] = State(
                            state_id, instance,
                            [mapping[fact] for fact in get_augmented_state(problem, node.state, self.config, action)])
                        self.node_id_to_state_ids[(self.problem_name_to_id[problem.name], node.id)][action] = state_id
                else:
                    state_id = next_state_id
                    next_state_id += 1
                    self.states[state_id] = State(
                        state_id, instance, [mapping[fact] for fact in get_goal_augmented_state(problem, node.state)])
                    self.node_id_to_state_ids[(self.problem_name_to_id[problem.name], node.id)] = state_id
        factory = SyntacticElementFactory(vocabulary)
        if config.get('preset_features', None):
            str_gens = config['preset_features']
        else:
            if all_generators:
                feature_generator_kwargs = config['unrestricted_feature_generator']
            else:
                feature_generator_kwargs = config['feature_generator']
            booleans, numericals, concepts, roles = dlplan_gen.generate_features(factory, list(self.states.values()),
                                                                                 *5 * [max_complexity], 3600, 10000,
                                                                                 **feature_generator_kwargs)
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

    def generate_augmented_state_space(self, problem):
        return generate_state_space(self.domain, problem)

    def evaluate_concept(self, concept, problem, state):
        return self.evaluate_concept_from_problem(concept, problem, state)

    def evaluate_role(self, role, problem, state):
        return self.evaluate_role_from_problem(role, problem, state)

    def evaluate_feature(self, feature, problem, state, action=None):
        return self.evaluate_feature_from_problem(feature, problem, state, action)

    def get_augmented_dlplan_state(self, problem, state, action=None):
        return State(
            -1, self.instances[problem.name],
            [self.mappings[problem.name][fact] for fact in get_augmented_state(problem, state, self.config, action)])

    def evaluate_feature_from_problem(self, feature, problem, state, action=None):
        feature = feature.strip('"')
        return self.features[feature].evaluate(self.get_augmented_dlplan_state(problem, state, action))

    def evaluate_role_from_problem(self, role, problem, state):
        role = role.strip('"')
        return set([(self.obj_id_to_obj(problem, id1), self.obj_id_to_obj(problem, id2)) for id1, id2 in
                    self.roles[role].evaluate(self.get_augmented_dlplan_state(problem, state)).to_sorted_vector()])

    def obj_id_to_obj(self, problem, obj_id):
        for obj in self.instances[problem.name].get_objects():
            if obj_id == obj.get_index():
                return obj.get_name()
        return None

    def evaluate_concept_from_problem(self, concept, problem, state):
        concept = concept.strip('"')
        return set([
            self.obj_id_to_obj(problem, id) for id in self.concepts[concept].evaluate(
                self.get_augmented_dlplan_state(problem, state)).to_sorted_vector()
        ])

    def is_concept_informative(self, concept_str):
        for problem, state_graph in self.state_graphs.items():
            for node in state_graph.nodes.values():
                extension = self.evaluate_concept_from_problem(concept_str, self.problems[problem], node.state)
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
                extension = self.evaluate_role_from_problem(role_str, self.problems[problem], node.state)
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

    def node_to_clingo(self, problem, node, stats):
        problem_id = self.problem_name_to_id[problem.name]
        clingo_program = ""
        clingo_program += f'state({problem_id}, {node.id}).\n'
        if node.alive == Alive.ALIVE:
            clingo_program += f'alive({problem_id}, {node.id}).\n'
        if node.alive != Alive.ALIVE and not self.config['include_dead_states']:
            return ""
        if check_formula(node.state, problem.goal):
            clingo_program += f'goal({problem_id}, {node.id}).\n'
        if self.config['include_actions']:
            for action, aug_state in self.node_id_to_state_ids[(problem_id, node.id)].items():
                action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
                clingo_program += f'aug_state({problem_id}, {node.id}, {action_str}, {aug_state}).\n'
                for feature_str, feature in self.features.items():
                    feature_str = f'"{feature_str}"'
                    eval = feature.evaluate(self.states[aug_state])
                    if type(eval) is bool:
                        eval = 1 if eval else 0
                    clingo_program += f'eval({aug_state}, {feature_str}, {eval}).\n'
                    stats['num_feature_evals'] += 1
        else:
            for feature_str, feature in self.features.items():
                feature_str = f'"{feature_str}"'
                eval = feature.evaluate(self.states[self.node_id_to_state_ids[(problem_id, node.id)]])
                if type(eval) is bool:
                    eval = 1 if eval else 0
                clingo_program += f'eval({problem_id}, {node.id}, {feature_str}, {eval}).\n'
                stats['num_feature_evals'] += 1
        all_action_args = {str(p) for action in node.children.keys() for p in action.parameters}
        for concept_str, concept in self.concepts.items():
            if concept_str in stats['uninformative_concepts']:
                #log.debug(f'Concept {concept_str} does not distinguish any action arguments, skipping')
                stats['num_skipped_concept_evals'] += len(
                    self.evaluate_concept_from_problem(f'"{concept_str}"', problem, node.state))
                continue
            concept_str = f'"{concept_str}"'
            extension = self.evaluate_concept_from_problem(concept_str, problem, node.state)
            for obj in extension:
                clingo_program += f'c_eval({problem_id}, {node.id}, {concept_str}, "{obj}").\n'
                stats['num_concept_evals'] += 1
        for role_str, role in self.roles.items():
            if role_str in stats['uninformative_roles']:
                #log.debug(f'Role {role_str} does not distinguish any action argument pairs, skipping')
                stats['num_skipped_role_evals'] += len(
                    self.evaluate_role_from_problem(f'"{role_str}"', problem, node.state))
                continue
            role_str = f'"{role_str}"'
            for obj1, obj2 in self.evaluate_role_from_problem(role_str, problem, node.state):
                clingo_program += f'r_eval({problem_id}, {node.id}, {role_str}, "{obj1}", "{obj2}").\n'
                stats['num_role_evals'] += 1
        for action, children in node.children.items():
            action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
            for child in children:
                clingo_program += f'trans({problem_id}, {node.id}, {action_str}, {child.id}).\n'
                if self.concepts:
                    params = [f'"{p}"' for p in action.parameters]
                    assert len(params) <= MAX_ACTION_PARAMETERS, \
                            f'Action {action.name} has too many parameters: {len(params)}'
                    if params:
                        clingo_program += f'amap({action_str}, "{action.name}", {", ".join(params)}).\n'
                    else:
                        clingo_program += f'amap({action_str}, "{action.name}").\n'
        return clingo_program

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
        stats = {
            'num_feature_evals': 0,
            'num_concept_evals': 0,
            'num_skipped_concept_evals': 0,
            'num_role_evals': 0,
            'num_skipped_role_evals': 0,
            'uninformative_concepts':
            self.compute_uninformative_concepts() if self.config['prune_concepts'] else set(),
            'uninformative_roles': self.compute_uninformative_roles() if self.config['prune_roles'] else set(),
        }
        for state_graph in self.state_graphs.values():
            for node in state_graph.nodes.values():
                clingo_program += self.node_to_clingo(state_graph.problem, node, stats)
        log.info(f'Generated program with {stats["num_feature_evals"]} feature evaluations, '
                 f'{stats["num_concept_evals"]} concept evaluations ({stats["num_skipped_concept_evals"]} skipped), '
                 f'and {stats["num_role_evals"]} role evaluations ({stats["num_skipped_role_evals"]} skipped)')
        return clingo_program
