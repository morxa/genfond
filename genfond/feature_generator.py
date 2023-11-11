from dlplan.core import VocabularyInfo, InstanceInfo, State, \
    SyntacticElementFactory
import dlplan.generator as dlplan_gen
from pddl.logic import Predicate
from .ground import ground_domain_predicates
from .state_space_generator import apply_effects, generate_state_space


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
    state = apply_effects(set(), goal_formula)
    goal_state = {
        Predicate(f'{predicate.name}_G', *predicate.terms)
        for predicate in state
    }
    return goal_state


def construct_instance_info(vocabulary, domain, problem):
    instance = InstanceInfo(vocabulary)
    map = dict()
    for object in problem.objects:
        instance.add_object(object.name)
    for predicate in ground_domain_predicates(domain, problem):
        map[predicate] = instance.add_atom(predicate.name,
                                           [str(t) for t in predicate.terms])
        goal_predicate = Predicate(f'{predicate.name}_G', *predicate.terms)
        map[goal_predicate] = instance.add_atom(
            goal_predicate.name, [str(t) for t in predicate.terms])
    return instance, map


class FeaturePool:

    def __init__(self, domain, problems):
        assert len({problem.name for problem in problems}) == len(problems), \
            "Problem names must be unique."
        vocabulary = construct_vocabulary_info(domain)
        self.states = dict()
        self.state_graphs = dict()
        self.goal_states = dict()
        for problem in problems:
            instance, mapping = construct_instance_info(
                vocabulary, domain, problem)
            self.state_graphs[problem.name] = generate_state_space(
                domain, problem)
            self.goal_states[problem.name] = _get_state_from_goal(problem.goal)
            pddl_states = {
                node.state | self.goal_states[problem.name]
                for node in self.state_graphs[problem.name].nodes.values()
            }
            self.states[problem.name] = {
                state: State(instance,
                             [mapping[predicate] for predicate in state])
                for state in pddl_states
            }
        factory = SyntacticElementFactory(vocabulary)
        str_features = dlplan_gen.generate_features(factory, [
            state for state in self.states[problem.name].values()
            for problem in problems
        ])
        self.features = {}
        for str_feature in str_features:
            if str_feature.startswith("b_"):
                feature = factory.parse_boolean(str_feature)
            elif str_feature.startswith("n_"):
                feature = factory.parse_numerical(str_feature)
            else:
                continue
            self.features[feature.compute_repr()] = feature

    def evaluate_feature(self, feature, state):
        # Try to guess which problem the state belongs to.
        problems = [
            problem for (problem, states) in self.states.items()
            if state | self.goal_states[problem] in states
        ]
        assert len(problems) == 1, \
            'Cannot determine which problem the state belongs to,' \
            f' found {len(problems)} matching problems'
        return self.evaluate_feature_from_problem(feature, problems[0], state)

    def evaluate_feature_from_problem(self, feature, problem, state):
        return self.features[feature].evaluate(
            self.states[problem][state | self.goal_states[problem]])
