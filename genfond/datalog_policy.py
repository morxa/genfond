import logging
import re

log = logging.getLogger(__name__)


class DatalogPolicyRule:

    def __init__(self, head, tail):
        match = re.search(r'^(\w+)\(([\w, ]+)\)$', head)
        if not match:
            raise ValueError(f'Invalid head: {head}')
        self.name = match.groups()[0]
        self.parameters = match.groups()[1].replace(' ', '').split(',')

        self.tail_by_parameter = {name: [] for name in self.parameters}
        for parameter, concept in tail or []:
            if parameter not in self.parameters:
                raise ValueError(f'Free variable {parameter} not in head parameters {self.parameters}!')
            self.tail_by_parameter[parameter].append(concept.replace(' ', ''))

    def __eq__(self, other):
        return self.head == other.head and self.tail == other.tail #TODO

    def __repr__(self):
        tail = []
        for parameter, concepts in self.tail_by_parameter.items():
            tail.extend([str(concept).replace("0", parameter) for concept in concepts])
        return f'{self.name}({', '.join(self.parameters)}){(' :- ' + ', '.join(tail)) if len(tail) > 0 else ''}.'

    def __hash__(self):
        return hash(repr(self))


class DatalogPolicy:

    def __init__(self, rules):
        self.rules = rules
        
    def __repr__(self):
        return f'Rules: {" ".join([str(r) for r in self.rules])}'
