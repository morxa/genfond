import re
from genfond.datalog_policy import DatalogPolicyRule, DatalogPolicy


def generate_datalog_policy(solution):
    args_to_vars = dict()
    conds = dict()
    for instance, state, action in solution['good_action']:
        # TODO copied from DatalogPolicyRule
        match = re.search(r'^([^(]+)\(([\w, ]+)\)$', action.strip('"'))
        if not match:
            raise ValueError(f'Invalid head: {action}')
        name = match.groups()[0]
        parameters = match.groups()[1].replace(' ', '').split(',')
        arg_to_var = dict()
        vars = ['X', 'Y', 'Z']
        for i, parameter in enumerate(parameters):
            if parameter not in arg_to_var:
                arg_to_var[i + 1] = vars.pop(0)
        args_to_vars[(instance, state, action)] = arg_to_var
    print(args_to_vars)
    rules = []
    for instance, state, action, _, _, _, concept, pos, argnum in solution['distinguished']:
        action = action.strip('"')
        concept = concept.strip('"')
        argnum = int(argnum)
        negated = (pos == 'neg')
        if (instance, state, action) not in conds:
            conds[(instance, state, action)] = []
        if concept == 'name':
            continue
        if negated:
            concept = f'c_not({concept})'
        var = args_to_vars[(instance, state, action)][argnum]
        cond = (var, concept)
        conds[(instance, state, action)].append(cond)
    for key, body_conds in conds.items():
        action = key[2]
        match = re.search(r'^([^(]+)\(([\w, ]+)\)$', action.strip('"'))
        if not match:
            raise ValueError(f'Invalid head: {action}')
        action_name = match.groups()[0]
        args = ",".join(args_to_vars[key].values())
        action = f'{action_name}({args})'
        rule = DatalogPolicyRule(action, body_conds)
        rules.append(rule)
    return DatalogPolicy(rules, cost=solution['cost'])
