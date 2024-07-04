import re
from genfond.datalog_policy import DatalogPolicyRule, DatalogPolicy, Cond, split_action_string
import logging

log = logging.getLogger(__name__)


def generate_datalog_policy(solution):
    args_to_vars = dict()
    conds = dict()
    for instance, state, action in solution['good_action']:
        name, parameters = split_action_string(action)
        arg_to_var = dict()
        # TODO do not hardcode the number of variables
        vars = ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        for i, parameter in enumerate(parameters):
            if parameter not in arg_to_var:
                arg_to_var[i + 1] = vars.pop(0)
        args_to_vars[(instance, state, action)] = arg_to_var
    bool_eval_dict = dict()
    for i, s, f, v in solution.get('bool_eval', []):
        bool_eval_dict[(i, s, f)] = v
    rules = []
    dist_features = dict()
    for instance, state, _, _, feature in solution.get('f_distinguished', []):
        dist_features.setdefault((instance, state), []).append(feature)
    state_conds = dict()
    for instance, state, action in solution['good_action']:
        conds[(instance, state, action)] = {'concepts': [], 'roles': []}
        state_cond = dict()
        for f in dist_features.get((instance, state), []):
            v = bool_eval_dict[(instance, state, f)]
            log.debug(f'Adding condition {f}={v}')
            if f.startswith('b_'):
                if v == 1:
                    state_cond[f] = Cond.TRUE
                elif v == 0:
                    state_cond[f] = Cond.FALSE
                else:
                    raise ValueError(f'Unknown value {v}')
            elif f.startswith('n_'):
                if v == 1:
                    state_cond[f] = Cond.POSITIVE
                elif v == 0:
                    state_cond[f] = Cond.ZERO
                else:
                    raise ValueError(f'Unknown value {v}')
            else:
                raise ValueError(f'Unknown value {v}')
        state_conds[(instance, state, action)] = state_cond
    log.debug(f'state_conds: {state_conds}')
    for instance, state, action, _, _, _, concept, pos, argnum in solution.get('c_distinguished', []):
        action = action.strip('"')
        concept = concept.strip('"')
        argnum = int(argnum)
        negated = (pos == 'neg')
        if concept == 'name':
            continue
        if negated:
            concept = f'c_not({concept})'
        var = args_to_vars[(instance, state, action)][argnum]
        cond = (var, concept)
        conds[(instance, state, action)]['concepts'].append(cond)
    for instance, state, action, _, _, _, role, pos, argnum1, argnum2 in solution.get('r_distinguished', []):
        action = action.strip('"')
        role = role.strip('"')
        argnum1 = int(argnum1)
        argnum2 = int(argnum2)
        negated = (pos == 'neg')
        if negated:
            role = f'r_not({role})'
        var1 = args_to_vars[(instance, state, action)][argnum1]
        var2 = args_to_vars[(instance, state, action)][argnum2]
        cond = (var1, var2, role)
        conds[(instance, state, action)]['roles'].append(cond)
    for key, cond_dict in conds.items():
        action = key[2]
        action_name, _ = split_action_string(action)
        args = ",".join(args_to_vars[key].values())
        action = f'{action_name}({args})'
        rule = DatalogPolicyRule(action,
                                 concepts=cond_dict['concepts'],
                                 roles=cond_dict['roles'],
                                 conds=state_conds[key])
        rules.append(rule)
    return DatalogPolicy(rules, cost=solution['cost'])
