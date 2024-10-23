import logging
import re
from .policy import Cond, cond_to_str
from frozendict import frozendict

log = logging.getLogger('genfond.policy.datalog')

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

    def __init__(self,
                 head,
                 concepts=None,
                 roles=None,
                 conds=None,
                 state_aug_conds=None,
                 param_aug_conds=None,
                 param_diff_conds=None):
        if not concepts:
            concepts = []
        if not roles:
            roles = []
        if not conds:
            conds = []
        if not state_aug_conds:
            state_aug_conds = []
        if not param_aug_conds:
            param_aug_conds = []
        if not param_diff_conds:
            param_diff_conds = []
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
        self.state_aug_conds = frozendict(state_aug_conds)
        self.param_aug_conds = frozendict(param_aug_conds)
        self.param_diff_conds = frozenset(param_diff_conds)

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if len(self.parameters) != len(other.parameters):
            return False
        if self.conds != other.conds:
            return False
        if self.state_aug_conds != other.state_aug_conds:
            return False
        if self.param_aug_conds != other.param_aug_conds:
            return False
        if self.param_diff_conds != other.param_diff_conds:
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
        state_cond_strs = [cond_to_str(cond, val) for cond, val in self.conds.items()]
        state_cond_strs.sort()
        state_aug_cond_strs = [cond_to_str(cond, val) for cond, val in self.state_aug_conds.items()]
        state_aug_cond_strs.sort()
        param_aug_cond_strs = [
            cond_to_str(cond, val[1], RULE_VARS[val[0]]) for cond, val in self.param_aug_conds.items()
        ]
        param_aug_cond_strs.sort()
        concept_cond_strs = []
        roles_cond_strs = []
        for parameter, concepts in self.concepts_by_parameter.items():
            concept_cond_strs.extend([f'{parameter} ∊ {concept}' for concept in concepts])
        concept_cond_strs.sort()
        for parameters, roles in self.roles_by_parameter.items():
            roles_cond_strs.extend([f'{parameters[0]} {role} {parameters[1]}' for role in roles])
        roles_cond_strs.sort()
        diff_cond_strs = []
        for feature, param1, param2, diff in self.param_diff_conds:
            if diff == 1:
                diff_cond_strs.append(f'{feature}({RULE_VARS[param1]},{RULE_VARS[param2]}) > 0')
            elif diff == -1:
                diff_cond_strs.append(f'{feature}({RULE_VARS[param1]},{RULE_VARS[param2]}) < 0')
            elif diff == 0:
                diff_cond_strs.append(f'{feature}({RULE_VARS[param1]},{RULE_VARS[param2]}) = 0')
            else:
                raise ValueError(f'Invalid diff value: {diff}')
        diff_cond_strs.sort()
        conds = state_cond_strs + state_aug_cond_strs + concept_cond_strs + roles_cond_strs + diff_cond_strs
        return f'{self.name}({', '.join(self.parameters)}){f" :- {', '.join(conds)}" if conds else ""}.'

    def __hash__(self):
        return hash((
            self.name,
            len(self.parameters),
            self.conds,
            self.state_aug_conds,
            self.param_aug_conds,
            self.param_diff_conds,
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
