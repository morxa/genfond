import logging
import re
from .policy import Cond, cond_to_str
from frozendict import frozendict

log = logging.getLogger(__name__)


class DatalogPolicyRule:

    def __init__(self, head, tail, conds=[]):
        match = re.search(r'^([^(]+)\(([\w, ]+)\)$', head)
        if not match:
            raise ValueError(f'Invalid head: {head}')
        self.name = match.groups()[0]
        self.parameters = match.groups()[1].replace(' ', '').split(',')

        tail_by_parameter = {name: [] for name in self.parameters}
        for parameter, concept in tail or []:
            if parameter not in self.parameters:
                raise ValueError(f'Free variable {parameter} not in head parameters {self.parameters}!')
            tail_by_parameter[parameter].append(concept.replace(' ', ''))
        self.tail_by_parameter = {name: frozenset(concepts) for name, concepts in tail_by_parameter.items()}
        self.conds = frozendict(conds)

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if len(self.parameters) != len(other.parameters):
            return False
        if self.conds != other.conds:
            return False
        for p_self, p_other in zip(self.parameters, other.parameters):
            if self.tail_by_parameter[p_self] != other.tail_by_parameter[p_other]:
                return False
        return True

    def __repr__(self):
        tail = [cond_to_str(cond, val) for cond, val in self.conds.items()]
        for parameter, concepts in self.tail_by_parameter.items():
            tail.extend([f'{parameter} ∊ {concept}' for concept in concepts])
        tail.sort()
        return f'{self.name}({', '.join(self.parameters)}){f" :- {', '.join(tail)}" if tail else ' '}.'

    def __hash__(self):
        return hash((
            self.name,
            len(self.parameters),
            self.conds,
            *tuple(self.tail_by_parameter[p] for p in self.parameters),
        ))


class DatalogPolicy:

    def __init__(self, rules, cost=0):
        self.cost = cost
        self.rules = frozenset(rules)

    def __eq__(self, other):
        return self.rules == other.rules

    def __repr__(self):
        return (f'{len(self.rules)} rules\n' + "\n".join([str(r) for r in self.rules]))
