from dlplan.core import VocabularyInfo, InstanceInfo, State, \
    SyntacticElementFactory
import dlplan.generator as dlplan_gen
from pddl.logic import Predicate
from .ground import ground_domain_predicates
from .state_space_generator import apply_effects, generate_state_space, check_formula, Alive

MAX_ACTION_PARAMETERS = 4


def construct_vocabulary_info(domain):
    vocabulary = VocabularyInfo()
    for predicate in domain.predicates:
        assert f'{predicate.name}_G' not in [p.name for p in domain.predicates]
    for predicate in domain.predicates:
        # TODO some predicates may be static.
        vocabulary.add_predicate(predicate.name, predicate.arity)
        vocabulary.add_predicate(f'{predicate.name}_G', predicate.arity)
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


def construct_instance_info(vocabulary, domain, problem, problem_id):
    instance = InstanceInfo(problem_id, vocabulary)
    map = dict()
    for object in problem.objects:
        instance.add_object(object.name)
    for predicate in ground_domain_predicates(domain, problem):
        map[predicate] = instance.add_atom(predicate.name, [str(t) for t in predicate.terms])
        goal_predicate = Predicate(f'{predicate.name}_G', *predicate.terms)
        map[goal_predicate] = instance.add_atom(goal_predicate.name, [str(t) for t in predicate.terms])
    return instance, map


class FeaturePool:

    def __init__(
        self,
        domain,
        problems,
        config,
        max_complexity=None,
        all_generators=True,
    ):
        assert len({problem.name for problem in problems}) == len(problems), \
            "Problem names must be unique."
        self.problem_name_to_id = {problem.name: i for i, problem in enumerate(problems)}
        vocabulary = construct_vocabulary_info(domain)
        self.states = dict()
        self.state_graphs = dict()
        self.goal_states = dict()
        self.instances = dict()
        if not max_complexity:
            max_complexity = config['max_complexity']
        for problem in problems:
            instance, mapping = construct_instance_info(vocabulary, domain, problem,
                                                        self.problem_name_to_id[problem.name])
            self.state_graphs[problem.name] = generate_state_space(domain, problem)
            self.goal_states[problem.name] = _get_state_from_goal(problem.goal)
            self.instances[problem.name] = instance
            pddl_states = {
                node.state | self.goal_states[problem.name]
                for node in self.state_graphs[problem.name].nodes.values()
            }
            self.states[problem.name] = {
                state: State(-1, instance, [mapping[predicate] for predicate in state])
                for state in pddl_states
            }
        factory = SyntacticElementFactory(vocabulary)
        if config.get('preset_features', None):
            str_gens = config['preset_features']
        else:
            str_gens = dlplan_gen.generate_features(
                factory, [state for state in self.states[problem.name].values() for problem in problems],
                *5 * [max_complexity], 3600, 10000, *30 * [True] if all_generators else [])
        self.features = {}
        self.concepts = {}
        self.roles = {}
        for str_gen in str_gens:
            if config['include_boolean_features'] and str_gen.startswith("b_"):
                feature = factory.parse_boolean(str_gen)
                self.features[str(feature)] = feature
            elif config['include_numerical_features'] and str_gen.startswith("n_"):
                feature = factory.parse_numerical(str_gen)
                self.features[str(feature)] = feature
            elif config['include_concepts'] and str_gen.startswith("c_"):
                self.concepts[str_gen] = factory.parse_concept(str_gen)
            elif config['include_roles'] and str_gen.startswith("r_"):
                self.roles[str_gen] = factory.parse_role(str_gen)

    def evaluate_feature(self, feature, state):
        # Try to guess which problem the state belongs to.
        problems = [
            problem for (problem, states) in self.states.items() if state | self.goal_states[problem] in states
        ]
        assert len(problems) == 1, \
            'Cannot determine which problem the state belongs to,' \
            f' found {len(problems)} matching problems'
        return self.evaluate_feature_from_problem(feature, problems[0], state)

    def evaluate_concept(self, concept, state):
        # Try to guess which problem the state belongs to.
        problems = [
            problem for (problem, states) in self.states.items() if state | self.goal_states[problem] in states
        ]
        assert len(problems) == 1, \
            'Cannot determine which problem the state belongs to,' \
            f' found {len(problems)} matching problems'
        return self.evaluate_concept_from_problem(concept, problems[0], state)

    def evaluate_role(self, role, state):
        # Try to guess which problem the state belongs to.
        problems = [
            problem for (problem, states) in self.states.items() if state | self.goal_states[problem] in states
        ]
        assert len(problems) == 1, \
            'Cannot determine which problem the state belongs to,' \
            f' found {len(problems)} matching problems'
        return self.evaluate_role_from_problem(role, problems[0], state)

    def evaluate_feature_from_problem(self, feature, problem, state):
        feature = feature.strip('"')
        return self.features[feature].evaluate(self.states[problem][state | self.goal_states[problem]])

    def evaluate_role_from_problem(self, role, problem, state):
        role = role.strip('"')
        return set([(self.obj_id_to_obj(problem, id1), self.obj_id_to_obj(problem, id2))
                    for id1, id2 in self.roles[role].evaluate(self.states[problem]
                                                              [state | self.goal_states[problem]]).to_sorted_vector()])

    def obj_id_to_obj(self, problem, obj_id):
        for obj in self.instances[problem].get_objects():
            if obj_id == obj.get_index():
                return obj.get_name()
        return None

    def evaluate_concept_from_problem(self, concept, problem, state):
        concept = concept.strip('"')
        return set([
            self.obj_id_to_obj(problem, id) for id in self.concepts[concept].evaluate(
                self.states[problem][state
                                     | self.goal_states[problem]]).to_sorted_vector()
        ])

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
        for problem, state_graph in self.state_graphs.items():
            problem_id = self.problem_name_to_id[problem]
            for node in state_graph.nodes.values():
                clingo_program += f'state({problem_id}, {node.id}).\n'
                if node.alive == Alive.ALIVE:
                    clingo_program += f'alive({problem_id}, {node.id}).\n'
                if check_formula(node.state, state_graph.problem.goal):
                    clingo_program += f'goal({problem_id}, {node.id}).\n'
                for feature_str, feature in self.features.items():
                    feature_str = f'"{feature_str}"'
                    eval = self.evaluate_feature_from_problem(feature_str, problem, node.state)
                    if type(eval) is bool:
                        eval = 1 if eval else 0
                    clingo_program += f'eval({problem_id}, {node.id}, {feature_str}, {eval}).\n'
                for concept_str, concept in self.concepts.items():
                    concept_str = f'"{concept_str}"'
                    for obj in self.evaluate_concept_from_problem(concept_str, problem, node.state):
                        clingo_program += f'c_eval({problem_id}, {node.id}, {concept_str}, "{obj}").\n'
                for role_str, role in self.roles.items():
                    role_str = f'"{role_str}"'
                    for obj1, obj2 in self.evaluate_role_from_problem(role_str, problem, node.state):
                        clingo_program += f'r_eval({problem_id}, {node.id}, {role_str}, "{obj1}", "{obj2}").\n'
                for action, children in node.children.items():
                    action_str = f'"{action.name}({",".join([str(p) for p in action.parameters])})"'
                    for child in children:
                        clingo_program += f'trans({problem_id}, {node.id}, {action_str}, {child.id}).\n'
                        if self.concepts:
                            params = [f'"{p}"' for p in action.parameters]
                            assert len(params) <= MAX_ACTION_PARAMETERS, \
                                    f'Action {action.name} has too many parameters: {len(params)}'
                            params += ['nil'] * (MAX_ACTION_PARAMETERS - len(params))
                            clingo_program += f'amap({action_str}, "{action.name}", {", ".join(params)}).\n'
        return clingo_program
