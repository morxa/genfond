import re
from genfond.datalog_policy import DatalogPolicyRule, DatalogPolicy, split_action_string, RULE_VARS
from genfond.policy import Cond, Effect
import logging

log = logging.getLogger(__name__)


def eval_to_cond(f, v):
    if f.startswith('b_'):
        if v == 1:
            return Cond.TRUE
        elif v == 0:
            return Cond.FALSE
        else:
            raise ValueError(f'Unknown value {v}')
    elif f.startswith('n_'):
        if v == 1:
            return Cond.POSITIVE
        elif v == 0:
            return Cond.ZERO
        else:
            raise ValueError(f'Unknown value {v}')
    else:
        raise ValueError(f'Unknown value {v}')


def generate_datalog_policy(solution):
    log.info(f'Generating policy from solution with {len(solution["good_action"])} good actions,'
             f' {len(solution.get("f_distinguished", []))} distinguished features,'
             f' {len(solution.get("c_distinguished", []))} distinguished concepts,'
             f' {len(solution.get("r_distinguished", []))} distinguished roles')
    args_to_vars = dict()
    conds = dict()
    for instance, state, action in solution['good_action']:
        log.debug(f'Good action {action} in state {state} of instance {instance}')
        name, parameters = split_action_string(action)
        arg_to_var = dict()
        vars = RULE_VARS.copy()
        for i, parameter in enumerate(parameters):
            if parameter not in arg_to_var:
                arg_to_var[i + 1] = vars.pop(0)
        args_to_vars[(instance, state, action)] = arg_to_var
    bool_eval_dict = dict()
    for i, s, f, v in solution.get('bool_eval', []):
        bool_eval_dict[(i, s, f)] = v
    aug_bool_eval_dict = dict()
    state_aug_bool_eval_dict = dict()
    for i, s, a, f, v in solution.get('state_aug_bool_eval', []):
        state_aug_bool_eval_dict[(i, s, a, f)] = v
    for i, s, a, p, f, v in solution.get('aug_bool_eval', []):
        aug_bool_eval_dict[(i, s, a, p, f)] = v
    rules = set()
    dist_features = dict()
    state_aug_dist_features = dict()
    param_aug_dist_features = dict()
    for instance, state, _, _, feature in solution.get('f_distinguished', []):
        dist_features.setdefault((instance, state), []).append(feature)
    for instance, state, action, _, _, _, feature in solution.get('state_aug_dist', []):
        state_aug_dist_features.setdefault((instance, state, action), []).append(feature)
    for instance, state, action, param, _, _, _, feature in solution.get('param_aug_dist', []):
        param_aug_dist_features.setdefault((instance, state, action), []).append((feature, param))
    diff_conds = dict()
    for instance, state, action, param1, param2, feature, diff in solution.get(f'fdiff', []):
        diff_conds.setdefault((instance, state, action), []).append((feature, param1, param2, diff))
    state_conds = dict()
    for instance, state, action in solution['good_action']:
        state_cond = dict()
        state_aug_cond = dict()
        param_aug_cond = dict()
        for f in dist_features.get((instance, state), []):
            v = bool_eval_dict[(instance, state, f)]
            log.debug(f'Adding state condition {f}={v}')
            state_cond[f] = eval_to_cond(f, v)
        state_conds[(instance, state, action)] = state_cond
        for f in state_aug_dist_features.get((instance, state, action), []):
            v = state_aug_bool_eval_dict[(instance, state, action, f)]
            log.debug(f'Adding state augmented condition {f}={v}')
            state_aug_cond[f] = eval_to_cond(f, v)
        for f, p in param_aug_dist_features.get((instance, state, action), []):
            v = aug_bool_eval_dict[(instance, state, action, p, f)]
            log.debug(f'Adding augmented condition {f}={v} for param {p}')
            param_aug_cond[f] = (p, eval_to_cond(f, v))
        diff_cond = []
        for f, param1, param2, v in diff_conds.get((instance, state, action), []):
            log.debug(f'Adding diff condition {f}({param1},{param2})={v}')
            diff_cond.append((f, param1, param2, v))
        conds[(instance, state, action)] = {
            'concepts': [],
            'roles': [],
            'state_aug_conds': state_aug_cond,
            'param_aug_conds': param_aug_cond,
            'diff_conds': diff_cond
        }
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
                                 conds=state_conds[key],
                                 state_aug_conds=cond_dict['state_aug_conds'],
                                 param_aug_conds=cond_dict['param_aug_conds'],
                                 param_diff_conds=cond_dict['diff_conds'])
        rules.add(rule)
    return DatalogPolicy(list(rules), cost=solution['cost'])
