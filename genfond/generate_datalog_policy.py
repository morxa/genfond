import re
from genfond.datalog_policy import DatalogPolicyRule, DatalogPolicy, Cond, ACTION_REGEX
import logging

log = logging.getLogger(__name__)


def generate_datalog_policy(solution):
    args_to_vars = dict()
    conds = dict()
    for instance, state, action in solution['good_action']:
        # TODO copied from DatalogPolicyRule
        match = re.search(ACTION_REGEX, action.strip('"'))
        if not match:
            raise ValueError(f'Invalid head: {action}')
        name = match.groups()[0]
        parameters = match.groups()[1].replace(' ', '').split(',')
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
        conds[(instance, state, action)] = []
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
        conds[(instance, state, action)].append(cond)
    for key, body_conds in conds.items():
        action = key[2]
        match = re.search(ACTION_REGEX, action.strip('"'))
        if not match:
            raise ValueError(f'Invalid head: {action}')
        action_name = match.groups()[0]
        args = ",".join(args_to_vars[key].values())
        action = f'{action_name}({args})'
        rule = DatalogPolicyRule(action, body_conds, state_conds[key])
        rules.append(rule)
    return DatalogPolicy(rules, cost=solution['cost'])
