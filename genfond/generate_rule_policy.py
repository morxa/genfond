from .policy import PolicyType
from .rule_policy import Policy, PolicyRule, Cond, Effect, StateConstraint
import logging

log = logging.getLogger('genfond.generation.rule')


def trans_deltas_to_effects(instance, state, trans_deltas):
    effs = dict()
    for i, s1, s2, f, v in trans_deltas:
        if i == instance and s1 == state:
            log.debug(f'Evaluating ({i}, {s1}, {s2}, {f}, {v}) for effects')
            eff = effs.setdefault(s2, set())
            if f.startswith('b_'):
                if v == 1:
                    eff.add((f, Effect.SET))
                    log.debug(f'Adding effect {f}={v}')
                elif v == -1:
                    eff.add((f, Effect.UNSET))
                    log.debug(f'Adding effect {f}={v}')
                elif v == 0:
                    continue
                else:
                    raise ValueError(f'Unknown value {v}')
            elif f.startswith('n_'):
                if v == 1:
                    eff.add((f, Effect.INCREASE))
                    log.debug(f'Adding effect {f}={v}')
                elif v == -1:
                    eff.add((f, Effect.DECREASE))
                    log.debug(f'Adding effect {f}={v}')
                elif v == 0:
                    continue
                else:
                    raise ValueError(f'Unknown value {v}')
            else:
                raise ValueError(f'Unknown value {v}')
    return effs


def feature_eval_to_cond(feature_str, feature_eval):
    log.debug(f'eval to cond: {feature_str} -> {feature_eval}')
    if feature_str.startswith("b_"):
        if feature_eval:
            return Cond.TRUE
        else:
            return Cond.FALSE
    elif feature_str.startswith("n_"):
        if feature_eval > 0:
            return Cond.POSITIVE
        else:
            return Cond.ZERO
    else:
        raise ValueError(f'Unknown feature type: {feature_str}')


def crit_states_to_constraints(crit_states, bool_eval):
    constraints = dict()
    for i, s, f, v in bool_eval:
        if (i, s) in crit_states:
            log.debug(f'Adding state constraint {f}={v}')
            constraints.setdefault((i, s), dict())[f] = feature_eval_to_cond(f, v)
    return {StateConstraint(conds) for conds in constraints.values()}


def generate_rule_policy(solution, policy_type=PolicyType.EXACT):
    rules = set()
    constraints = set()
    state_constraints = set()
    features = solution['selected']
    for instance, state in solution['state']:
        conds = dict()
        for i, s, f, v in solution['bool_eval']:
            if i == instance and s == state:
                log.debug(f'Adding condition {f}={v}')
                if f.startswith('b_'):
                    if v == 1:
                        conds[f] = Cond.TRUE
                    elif v == 0:
                        conds[f] = Cond.FALSE
                    else:
                        raise ValueError(f'Unknown value {v}')
                elif f.startswith('n_'):
                    if v == 1:
                        conds[f] = Cond.POSITIVE
                    elif v == 0:
                        conds[f] = Cond.ZERO
                    else:
                        raise ValueError(f'Unknown value {v}')
                else:
                    raise ValueError(f'Unknown value {v}')
        effs = trans_deltas_to_effects(instance, state, solution['good_trans_delta'])
        if effs:
            log.debug(f'Effects: {effs}')
            rule = PolicyRule(conds, frozenset({frozenset(e) for e in effs.values()}))
            log.debug(f'Adding rule {rule}')
            rules.add(rule)
        bad_effs = trans_deltas_to_effects(instance, state, solution.get('bad_trans_delta', []))
        if bad_effs:
            assert policy_type == PolicyType.CONSTRAINED, 'Cannot add constraints to exact policy'
            log.debug(f'Constraints: {bad_effs}')
            constraint = PolicyRule(conds, frozenset({frozenset(e) for e in bad_effs.values()}))
            log.debug(f'Adding constraint {constraint}')
            constraints.add(constraint)
    log.debug(f'Crit states: {solution.get("crit_state", [])}')
    state_constraints = crit_states_to_constraints(solution.get('crit_state', []), solution['bool_eval'])
    policy = Policy(features,
                    rules,
                    type=policy_type,
                    cost=solution['cost'],
                    constraints=constraints,
                    state_constraints=state_constraints)
    before_pruning_rules = len(policy.rules)
    before_pruning_state_constraints = len(policy.state_constraints)
    policy.simplify()
    log.info(f'Pruned {before_pruning_rules - len(policy.rules)} rules and '
             f'{before_pruning_state_constraints - len(policy.state_constraints)} state constraints')
    return policy
