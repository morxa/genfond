import logging
import re
from .policy import Cond, cond_to_str
from frozendict import frozendict

log = logging.getLogger(__name__)

ACTION_REGEX = r'^([^(]+)\(([^(]*)\)$'


def split_action_string(action):
    match = re.search(ACTION_REGEX, action.strip('"'))
    if not match:
        raise ValueError(f'Invalid action string: {action}')
    name = match.groups()[0]
    if match.groups()[1]:
        parameters = match.groups()[1].replace(' ', '').split(',')
    else:
        parameters = []
    return name, parameters


class DatalogPolicyRule:

    def __init__(self, head, concepts, conds=None):
        if not conds:
            conds = []
        self.name, self.parameters = split_action_string(head)
        concepts_by_parameter = {name: [] for name in self.parameters}
        for parameter, concept in concepts or []:
            if parameter not in self.parameters:
                raise ValueError(f'Free variable {parameter} not in head parameters {self.parameters}!')
            concepts_by_parameter[parameter].append(concept.replace(' ', ''))
        self.concepts_by_parameter = {name: frozenset(concepts) for name, concepts in concepts_by_parameter.items()}
        self.conds = frozendict(conds)

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if len(self.parameters) != len(other.parameters):
            return False
        if self.conds != other.conds:
            return False
        for p_self, p_other in zip(self.parameters, other.parameters):
            if self.concepts_by_parameter[p_self] != other.concepts_by_parameter[p_other]:
                return False
        return True

    def __repr__(self):
        state_conds = [cond_to_str(cond, val) for cond, val in self.conds.items()]
        state_conds.sort()
        concept_conds = []
        for parameter, concepts in self.concepts_by_parameter.items():
            concept_conds.extend([f'{parameter} ∊ {concept}' for concept in concepts])
        concept_conds.sort()
        conds = state_conds + concept_conds
        return f'{self.name}({', '.join(self.parameters)}){f" :- {', '.join(conds)}" if concepts else ' '}.'

    def __hash__(self):
        return hash((
            self.name,
            len(self.parameters),
            self.conds,
            *tuple(self.concepts_by_parameter[p] for p in self.parameters),
        ))


class DatalogPolicy:

    def __init__(self, rules, cost=0):
        self.cost = cost
        self.rules = frozenset(rules)

    def __eq__(self, other):
        return self.rules == other.rules

    def __repr__(self):
        return (f'{len(self.rules)} rules\n' + "\n".join([str(r) for r in self.rules]))

    def __hash__(self):
        return hash(self.rules)
