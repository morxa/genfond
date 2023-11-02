from dlplan.core import VocabularyInfo, InstanceInfo, State, \
    SyntacticElementFactory
import dlplan.generator as dlplan_gen
from genfond.ground import ground_domain_predicates


def construct_vocabulary_info(domain):
    vocabulary = VocabularyInfo()
    for predicate in domain.predicates:
        # TODO some predicates may be static.
        vocabulary.add_predicate(predicate.name, predicate.arity)
    for constant in domain.constants:
        vocabulary.add_constant(constant.name)
    return vocabulary


def construct_instance_info(vocabulary, domain, problem):
    instance = InstanceInfo(vocabulary)
    map = dict()
    for predicate in ground_domain_predicates(domain, problem):
        map[predicate] = instance.add_atom(predicate.name,
                                           [str(t) for t in predicate.terms])
    return instance, map


class FeaturePool:

    def __init__(self, domain, problem, pddl_states):
        vocabulary = construct_vocabulary_info(domain)
        instance, mapping = construct_instance_info(vocabulary, domain,
                                                    problem)
        factory = SyntacticElementFactory(vocabulary)
        self.states = {
            state: State(instance, [mapping[predicate] for predicate in state])
            for state in pddl_states
        }
        str_features = dlplan_gen.generate_features(factory,
                                                    list(self.states.values()))
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
        return self.features[feature].evaluate(self.states[state])
