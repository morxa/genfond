import logging
import re

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

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if len(self.parameters) != len(other.parameters):
            return False

        for p_self, p_other in zip(self.parameters, other.parameters):
            if self.tail_by_parameter[p_self] != other.tail_by_parameter[p_other]:
                return False

        return True

    def __repr__(self):
        tail = []
        for parameter, concepts in self.tail_by_parameter.items():
            tail.extend([f'{parameter} ∈ {concept}' for concept in concepts])
        tail.sort()
        return f'{self.name}({', '.join(self.parameters)}){f" :- {', '.join(tail)}" if tail else ' '}.'

    def __hash__(self):
        return hash((
            self.name,
            len(self.parameters),
            *tuple(self.tail_by_parameter[p] for p in self.parameters),
        ))


class DatalogPolicy:

    def __init__(self, rules, cost=0):
        self.cost = cost
        self.rules = frozenset(rules)

    def __eq__(self, other):
        return self.rules == other.rules

    def __repr__(self):
        return "\n".join([str(r) for r in self.rules])
