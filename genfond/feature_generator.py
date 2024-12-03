from dlplan.core import VocabularyInfo, InstanceInfo, State, \
    SyntacticElementFactory
import dlplan.generator as dlplan_gen
from pddl.logic import Predicate
from .ground import ground_domain_predicates, ground
from .state_space_generator import apply_effects, generate_state_space, check_formula, Alive
import logging
import itertools

log = logging.getLogger('genfond.feature_generation')

MAX_ACTION_PARAMETERS = 4


def get_aparam_predicate_name(i):
    return f'aparam{i}'


def get_aparam_predicate(i, param):
    return Predicate(get_aparam_predicate_name(i), param)


def construct_vocabulary_info(domain, config):
    vocabulary = VocabularyInfo()
    for predicate in domain.predicates:
        assert f'{predicate.name}_G' not in [p.name for p in domain.predicates]
    for predicate in domain.predicates:
        # TODO some predicates may be static.
        vocabulary.add_predicate(predicate.name, predicate.arity)
        vocabulary.add_predicate(f'{predicate.name}_G', predicate.arity)
    max_arity = max([len(action.parameters) for action in domain.actions])
    if config['include_actions'] or config['include_action_params']:
        for i in range(max_arity):
            #log.debug(f'Adding action parameter predicate {get_aparam_predicate_name(i)}')
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


def construct_instance_info(vocabulary, domain, problem, problem_id, config):
    instance = InstanceInfo(problem_id, vocabulary)
    map = dict()
    for object in problem.objects:
        instance.add_object(object.name)
    for predicate in ground_domain_predicates(domain, problem):
        map[predicate] = instance.add_atom(predicate.name, [str(t) for t in predicate.terms])
        goal_predicate = Predicate(f'{predicate.name}_G', *predicate.terms)
        map[goal_predicate] = instance.add_atom(goal_predicate.name, [str(t) for t in predicate.terms])
    if config['include_actions']:
        for action in ground(domain, problem):
            for i, param in enumerate(action.parameters):
                map[get_aparam_predicate(i, param)] = instance.add_atom(get_aparam_predicate_name(i), [str(param)])
    if config['include_action_params']:
        for action in ground(domain, problem):
            for i, param in enumerate(action.parameters):
                map[get_aparam_predicate(0, param)] = instance.add_atom(get_aparam_predicate_name(0), [str(param)])
    return instance, map


def get_goal_augmented_state(problem, state):
    return frozenset(state | _get_state_from_goal(problem.goal))


def get_param_augmented_state(problem, state, param_index, param):
    augmented_state = set(get_goal_augmented_state(problem, state))
    augmented_state.add(get_aparam_predicate(0, param))  # Always use 0 as we do not need the index
    #action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
    #log.debug(f'Generated augmented state (action={action_str}): [{", ".join([str(p) for p in augmented_state])}]')
    return frozenset(augmented_state)


def get_action_augmented_state(problem, state, config, action=None):
    assert not config['include_actions'] or action, 'Action must be provided when including actions'
    augmented_state = get_goal_augmented_state(problem, state)
    if config['include_actions']:
        param_atoms = {get_aparam_predicate(i, action.parameters[i]) for i, _ in enumerate(action.parameters)}
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
        selected_states=None,
    ):
        assert len({problem.name for problem in problems}) == len(problems), \
            "Problem names must be unique."
        self.domain = domain
        self.problems = {problem.name: problem for problem in problems}
        self.config = config
        self.problem_name_to_id = {problem.name: i for i, problem in enumerate(problems)}
        vocabulary = construct_vocabulary_info(domain, config)
        log.debug(f'Constructed vocabulary: {vocabulary}')
        self.states = dict()
        self.node_id_to_state_ids = dict()
        self.node_id_to_aug_state_ids = dict()
        self.state_id_to_node = dict()
        self.state_graphs = dict()
        self.instances = dict()
        self.mappings = dict()
        self.next_state_id = 0
        if not max_complexity:
            max_complexity = config['max_complexity']
        for problem in problems:
            instance, mapping = construct_instance_info(vocabulary, domain, problem,
                                                        self.problem_name_to_id[problem.name], config)
            self.state_graphs[problem.name] = generate_state_space(domain,
                                                                   problem,
                                                                   selected_states=selected_states[problem.name])
            self.instances[problem.name] = instance
            self.mappings[problem.name] = mapping
            for node in self.state_graphs[problem.name].nodes.values():
                if config['include_actions']:
                    self.node_to_action_augmented_state(problem, node)
                elif config['include_action_params']:
                    self.node_to_param_augmented_state(problem, node)
                if config['include_pristine_states']:
                    self.node_to_state(problem, node)
        factory = SyntacticElementFactory(vocabulary)
        if config.get('preset_features', None):
            booleans = [factory.parse_boolean(f) for f in config['preset_features'].get('booleans', [])]
            numericals = [factory.parse_numerical(f) for f in config['preset_features'].get('numericals', [])]
            concepts = [factory.parse_concept(c) for c in config['preset_features'].get('concepts', [])]
            roles = [factory.parse_role(r) for r in config['preset_features'].get('roles', [])]
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

    def node_to_action_augmented_state(self, problem, node):
        self.node_id_to_aug_state_ids.setdefault((self.problem_name_to_id[problem.name], node.id), dict())
        for action in node.children.keys():
            fstate = get_action_augmented_state(problem, node.state, self.config, action)
            state_id = self.next_state_id
            self.next_state_id += 1
            self.states[fstate] = State(state_id, self.instances[problem.name],
                                        [self.mappings[problem.name][fact] for fact in fstate])
            self.node_id_to_aug_state_ids[(self.problem_name_to_id[problem.name], node.id)][action] = fstate
            self.state_id_to_node.setdefault(fstate, []).append(node)

    def node_to_param_augmented_state(self, problem, node):
        self.node_id_to_aug_state_ids.setdefault((self.problem_name_to_id[problem.name], node.id), dict())
        for action in node.children.keys():
            self.node_id_to_aug_state_ids[(self.problem_name_to_id[problem.name], node.id)][action] = []
            for i, param in enumerate(action.parameters):
                fstate = get_param_augmented_state(problem, node.state, i, param)
                if fstate not in self.states:
                    state_id = self.next_state_id
                    self.next_state_id += 1
                    self.states[fstate] = State(state_id, self.instances[problem.name],
                                                [self.mappings[problem.name][fact] for fact in fstate])
                self.node_id_to_aug_state_ids[(self.problem_name_to_id[problem.name], node.id)][action].append(fstate)
                self.state_id_to_node.setdefault(fstate, []).append(node)

    def node_to_state(self, problem, node):
        state_id = self.next_state_id
        self.next_state_id += 1
        fstate = get_goal_augmented_state(problem, node.state)
        self.states[fstate] = State(state_id, self.instances[problem.name],
                                    [self.mappings[problem.name][fact] for fact in fstate])
        self.node_id_to_state_ids[(self.problem_name_to_id[problem.name], node.id)] = fstate
        self.state_id_to_node.setdefault(fstate, []).append(node)

    def generate_augmented_state_space(self, problem):
        return generate_state_space(self.domain, problem)

    def evaluate_concept(self, concept, problem, state):
        return self.evaluate_concept_from_problem(concept, problem, state)

    def evaluate_role(self, role, problem, state):
        return self.evaluate_role_from_problem(role, problem, state)

    def evaluate_feature(self, feature, problem, state, action=None):
        return self.evaluate_feature_from_problem(feature, problem, state, action)

    def get_augmented_dlplan_state(self, problem, state, action=None):
        return State(-1, self.instances[problem.name], [
            self.mappings[problem.name][fact]
            for fact in get_action_augmented_state(problem, state, self.config, action)
        ])

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
                if node.goal and not self.config['include_goal_states']:
                    continue
                if node.alive != Alive.ALIVE and not self.config['include_dead_states']:
                    continue
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
        log.info(f'Found {len(uninformative_concepts)} uninformative concept(s)')
        log.debug(", ".join(uninformative_concepts))
        return uninformative_concepts

    def is_role_informative(self, role_str):
        for problem, state_graph in self.state_graphs.items():
            for node in state_graph.nodes.values():
                if node.goal and not self.config['include_goal_states']:
                    continue
                if node.alive != Alive.ALIVE and not self.config['include_dead_states']:
                    continue
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
        log.info(f'Found {len(uninformative_roles)} uninformative role(s)')
        log.debug(", ".join(uninformative_roles))
        return uninformative_roles

    def is_feature_informative(self, feature):
        has_false = False
        has_true = False
        for fstate, state in self.states.items():
            if all([
                    node.goal and not self.config['include_goal_states']
                    or node.alive != Alive.ALIVE and not self.config['include_dead_states']
                    for node in self.state_id_to_node[fstate]
            ]):
                continue
            eval = feature.evaluate(state)
            if eval:
                has_true = True
            else:
                has_false = True
            if has_true and has_false:
                return True
        return has_true and has_false

    def compute_uninformative_features(self):
        uninformative_features = set()
        for feature_str, feature in self.features.items():
            if not self.is_feature_informative(feature):
                uninformative_features.add(feature_str)
        log.info(f'Found {len(uninformative_features)} uninformative feature(s)')
        log.debug(", ".join(uninformative_features))
        return uninformative_features

    def node_to_clingo(self, problem, node, stats):
        problem_id = self.problem_name_to_id[problem.name]
        clingo_program = ""
        clingo_program += f'% ' + ','.join([f'{p.name}({",".join([str(p) for p in p.terms])})'
                                            for p in node.state]) + '\n'
        clingo_program += f'state({problem_id}, {node.id}).\n'
        if node.alive == Alive.PRUNED:
            clingo_program += f'pruned({problem_id}, {node.id}).\n'
            return clingo_program
        if node.alive == Alive.ALIVE:
            clingo_program += f'alive({problem_id}, {node.id}).\n'
        if node.alive != Alive.ALIVE and not self.config['include_dead_states']:
            return clingo_program
        if check_formula(node.state, problem.goal):
            clingo_program += f'goal({problem_id}, {node.id}).\n'
            if not self.config['include_goal_states']:
                return clingo_program
        if self.config['include_actions']:
            for action, aug_state in self.node_id_to_aug_state_ids[(problem_id, node.id)].items():
                aug_state_id = self.states[aug_state].get_index()
                action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
                clingo_program += f'aug_state({problem_id}, {node.id}, {action_str}, {aug_state_id}).\n'
                for feature_str, feature in self.features.items():
                    if feature_str in stats['uninformative_features']:
                        stats['num_skipped_feature_evals'] += 1
                        continue
                    feature_str = f'"{feature_str}"'
                    eval = feature.evaluate(self.states[aug_state])
                    if type(eval) is bool:
                        eval = 1 if eval else 0
                    clingo_program += f'eval({aug_state_id}, {feature_str}, {eval}).\n'
                    stats['num_feature_evals'] += 1
        if self.config['include_pristine_states']:
            for feature_str, feature in self.features.items():
                if feature_str in stats['uninformative_features']:
                    stats['num_skipped_feature_evals'] += 1
                    continue
                feature_str = f'"{feature_str}"'
                eval = feature.evaluate(self.states[self.node_id_to_state_ids[(problem_id, node.id)]])
                if type(eval) is bool:
                    eval = 1 if eval else 0
                clingo_program += f'eval({problem_id}, {node.id}, {feature_str}, {eval}).\n'
                stats['num_feature_evals'] += 1
        if self.config['include_action_params']:
            for action, aug_states in self.node_id_to_aug_state_ids[(problem_id, node.id)].items():
                action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
                for i, aug_state in enumerate(aug_states):
                    aug_state_id = self.states[aug_state].get_index()
                    clingo_program += f'aug_state({problem_id}, {node.id}, {action_str}, {i}, {aug_state_id}).\n'
                    for feature_str, feature in self.features.items():
                        if feature_str in stats['uninformative_features']:
                            stats['num_skipped_feature_evals'] += 1
                            continue
                        if not get_aparam_predicate_name(0) in feature_str:
                            continue
                        feature_str = f'"{feature_str}"'
                        eval = feature.evaluate(self.states[aug_state])
                        if type(eval) is bool:
                            eval = 1 if eval else 0
                        clingo_program += f'aug_eval({aug_state_id}, {feature_str}, {eval}).\n'
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
                if obj in all_action_args:
                    clingo_program += f'c_eval({problem_id}, {node.id}, {concept_str}, "{obj}").\n'
                    stats['num_concept_evals'] += 1
                else:
                    stats['num_skipped_concept_evals'] += 1
        for role_str, role in self.roles.items():
            if role_str in stats['uninformative_roles']:
                #log.debug(f'Role {role_str} does not distinguish any action argument pairs, skipping')
                stats['num_skipped_role_evals'] += len(
                    self.evaluate_role_from_problem(f'"{role_str}"', problem, node.state))
                continue
            role_str = f'"{role_str}"'
            for obj1, obj2 in self.evaluate_role_from_problem(role_str, problem, node.state):
                if obj1 in all_action_args and obj2 in all_action_args:
                    clingo_program += f'r_eval({problem_id}, {node.id}, {role_str}, "{obj1}", "{obj2}").\n'
                    stats['num_role_evals'] += 1
                else:
                    stats['num_skipped_role_evals'] += 1
        for action, children in node.children.items():
            action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
            clingo_program += f'aname({action_str}, "{action.name}").\n'
            for child in children:
                clingo_program += f'trans({problem_id}, {node.id}, {action_str}, {child.id}).\n'
                if self.concepts:
                    params = [f'"{p}"' for p in action.parameters]
                    for i, p in enumerate(params):
                        clingo_program += f'aparam({action_str}, {i}, {p}).\n'
        return clingo_program

    def to_clingo(self):
        stats = {
            'num_skipped_feature_evals': 0,
            'num_feature_evals': 0,
            'num_concept_evals': 0,
            'num_skipped_concept_evals': 0,
            'num_role_evals': 0,
            'num_skipped_role_evals': 0,
            'uninformative_features':
            self.compute_uninformative_features() if self.config['prune_features'] else set(),
            'uninformative_concepts':
            self.compute_uninformative_concepts() if self.config['prune_concepts'] else set(),
            'uninformative_roles': self.compute_uninformative_roles() if self.config['prune_roles'] else set(),
        }
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
        for state_graph in self.state_graphs.values():
            for node in state_graph.nodes.values():
                clingo_program += self.node_to_clingo(state_graph.problem, node, stats)
        log.info(
            f'Generated program with {stats["num_feature_evals"]} feature evaluations ({stats["num_skipped_feature_evals"]} skipped), '
            f'{stats["num_concept_evals"]} concept evaluations ({stats["num_skipped_concept_evals"]} skipped), '
            f'and {stats["num_role_evals"]} role evaluations ({stats["num_skipped_role_evals"]} skipped)')
        return clingo_program
