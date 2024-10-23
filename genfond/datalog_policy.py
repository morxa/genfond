import logging
import re
from .policy import Cond, cond_to_str
from frozendict import frozendict

log = logging.getLogger(__name__)

ACTION_REGEX = r'^([^(]+)\(([^(]*)\)$'
RULE_VARS = ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


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

    def __init__(self, head, concepts=None, roles=None, conds=None, aug_conds=None, diff_conds=None):
        if not concepts:
            concepts = []
        if not roles:
            roles = []
        if not conds:
            conds = []
        if not aug_conds:
            aug_conds = []
        if not diff_conds:
            diff_conds = []
        self.name, self.parameters = split_action_string(head)
        concepts_by_parameter = {name: [] for name in self.parameters}
        roles_by_parameter = {}
        for parameter, concept in concepts or []:
            if parameter not in self.parameters:
                raise ValueError(f'Free variable {parameter} not in head parameters {self.parameters}!')
            concepts_by_parameter[parameter].append(concept.replace(' ', ''))
        for param1, param2, role in roles:
            if param1 not in self.parameters:
                raise ValueError(f'Free variable {param1} not in head parameters {self.parameters}!')
            if param2 not in self.parameters:
                raise ValueError(f'Free variable {param2} not in head parameters {self.parameters}!')
            roles_by_parameter.setdefault((param1, param2), []).append(role)
        self.concepts_by_parameter = {name: frozenset(concepts) for name, concepts in concepts_by_parameter.items()}
        self.roles_by_parameter = {params: frozenset(roles) for params, roles in roles_by_parameter.items()}
        self.conds = frozendict(conds)
        self.aug_conds = frozendict(aug_conds)
        self.diff_conds = diff_conds

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if len(self.parameters) != len(other.parameters):
            return False
        if self.conds != other.conds:
            return False
        if self.aug_conds != other.aug_conds:
            return False
        if self.diff_conds != other.diff_conds:
            return False
        for p_self, p_other in zip(self.parameters, other.parameters):
            if self.concepts_by_parameter[p_self] != other.concepts_by_parameter[p_other]:
                return False
        if self.roles_by_parameter.keys() != other.roles_by_parameter.keys():
            return False
        for params in self.roles_by_parameter.keys():
            if self.roles_by_parameter[params] != other.roles_by_parameter[params]:
                return False
        return True

    def __repr__(self):
        state_conds = [cond_to_str(cond, val) for cond, val in self.conds.items()]
        aug_state_conds = [cond_to_str(cond, val[1], RULE_VARS[val[0]]) for cond, val in self.aug_conds.items()]
        state_conds.sort()
        concept_conds = []
        roles_conds = []
        for parameter, concepts in self.concepts_by_parameter.items():
            concept_conds.extend([f'{parameter} ∊ {concept}' for concept in concepts])
        concept_conds.sort()
        for parameters, roles in self.roles_by_parameter.items():
            roles_conds.extend([f'{parameters[0]} {role} {parameters[1]}' for role in roles])
        roles_conds.sort()
        diff_conds = []
        for feature, param1, param2, diff in self.diff_conds:
            if diff == 1:
                diff_conds.append(f'{feature}({RULE_VARS[param1]},{RULE_VARS[param2]}) > 0')
            elif diff == -1:
                diff_conds.append(f'{feature}({RULE_VARS[param1]},{RULE_VARS[param2]}) < 0')
            elif diff == 0:
                diff_conds.append(f'{feature}({RULE_VARS[param1]},{RULE_VARS[param2]}) = 0')
            else:
                raise ValueError(f'Invalid diff value: {diff}')
        diff_conds.sort()
        conds = state_conds + aug_state_conds + concept_conds + roles_conds + diff_conds
        return f'{self.name}({', '.join(self.parameters)}){f" :- {', '.join(conds)}" if conds else ""}.'

    def __hash__(self):
        return hash((
            self.name,
            len(self.parameters),
            self.conds,
            self.aug_conds,
            *tuple(self.concepts_by_parameter[p] for p in self.parameters),
            *tuple((params, roles) for params, roles in self.roles_by_parameter.items()),
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
