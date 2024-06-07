import itertools
from enum import Enum
import logging
from .generate_datalog_policy import generate_datalog_policy

log = logging.getLogger(__name__)


class Cond(Enum):
    TRUE = 1
    FALSE = 2
    POSITIVE = 3
    ZERO = 4


class Effect(Enum):
    INCREASE = 1
    DECREASE = 2
    SET = 3
    UNSET = 4


class PolicyType(Enum):
    EXACT = 1
    CONSTRAINED = 2
    DATALOG = 3


class PolicyRule:

    def __init__(self, conds, effs):
        self.conds = conds
        self.effs = frozenset({frozenset(eff) for eff in effs})

    def __eq__(self, other):
        return self.conds == other.conds and self.effs == other.effs

    def __repr__(self):
        s_conds = []
        for feature, val in self.conds.items():
            if val == Cond.TRUE:
                s_conds.append(f'{feature}')
            elif val == Cond.FALSE:
                s_conds.append(f'¬{feature}')
            elif val == Cond.POSITIVE:
                s_conds.append(f'{feature} > 0')
            elif val == Cond.ZERO:
                s_conds.append(f'{feature} = 0')
            else:
                raise ValueError(f'Unknown feature type {feature}')
        s_effs = []
        for eff in self.effs:
            s_eff = []
            for feature, val in eff:
                if val == Effect.SET:
                    s_eff.append(f'{feature}')
                elif val == Effect.UNSET:
                    s_eff.append(f'¬{feature}')
                elif val == Effect.INCREASE:
                    s_eff.append(f'↑{feature}')
                elif val == Effect.DECREASE:
                    s_eff.append(f'↓{feature}')
                else:
                    raise ValueError(f'Unknown feature type {feature}')
            s_effs.append(' ∧ '.join(s_eff))
        return f'{{ {" ∧ ".join(sorted(s_conds))} }}  ⇒  {{ {" ; ".join(sorted(s_effs))} }}'

    def __hash__(self):
        return hash(repr(self))


class StateConstraint:

    def __init__(self, conds):
        self.conds = conds

    def __eq__(self, other):
        return self.conds == other.conds

    def __repr__(self):
        s_conds = []
        for feature, val in self.conds.items():
            if val == Cond.TRUE:
                s_conds.append(f'{feature}')
            elif val == Cond.FALSE:
                s_conds.append(f'¬{feature}')
            elif val == Cond.POSITIVE:
                s_conds.append(f'{feature} > 0')
            elif val == Cond.ZERO:
                s_conds.append(f'{feature} = 0')
            else:
                raise ValueError(f'Unknown feature type {feature}')
        return f'{{ {" ∧ ".join(sorted(s_conds))} }}'

    def __hash__(self):
        return hash(repr(self))


class Policy:

    def __init__(self, features, rules, cost=None, constraints=None, state_constraints=None, type=PolicyType.EXACT):
        self.type = type
        self.features = frozenset(features)
        self.rules = frozenset(rules)
        self.constraints = constraints or set()
        self.state_constraints = state_constraints or set()
        self.cost = cost
        assert not constraints or type == PolicyType.CONSTRAINED

    def __repr__(self):
        if self.type == PolicyType.CONSTRAINED:
            s_constraints = ''
            s_state_constraints = ''
            if self.constraints:
                s_constraints = '\nConstraints:\n' + '\n'.join(
                    sorted({repr(constraint)
                            for constraint in self.constraints}))
            if self.state_constraints:
                s_state_constraints = '\nState Constraints:\n' + '\n'.join(
                    sorted({repr(constraint)
                            for constraint in self.state_constraints}))

            return ('{} rules,'
                    ' {} transition constraints,'
                    ' {} state constraints with {} features: {}'
                    '\nRules:\n{}{}{}').format(len(self.rules), len(self.constraints), len(self.state_constraints),
                                               len(self.features), ", ".join(sorted(self.features)),
                                               '\n'.join(sorted({repr(rule)
                                                                 for rule in self.rules})), s_constraints,
                                               s_state_constraints)
        else:
            return '{} rules with {} features: {}\n{}'.format(len(self.rules), len(self.features),
                                                              ", ".join(sorted(self.features)),
                                                              '\n'.join(sorted({repr(rule)
                                                                                for rule in self.rules})))

    def simplify(self):
        self.simplify_rules()
        self.simplify_state_constraints()

    def simplify_rules(self):
        new_rules = set()
        pruned_rules = set()
        for r1, r2 in itertools.combinations(self.rules, 2):
            if r1.effs == r2.effs:
                if r1.conds.keys() < r2.conds.keys() and all([r2.conds[k] == v for k, v in r1.conds.items()]):
                    # One condition is a subset of the other, keep the less restrictive one
                    pruned_rules.add(r2)
                    continue
                elif r1.conds.keys() > r2.conds.keys() and all([r1.conds[k] == v for k, v in r2.conds.items()]):
                    # One condition is a subset of the other, keep the less restrictive one
                    pruned_rules.add(r1)
                    continue
                if r1.conds.keys() ^ r2.conds.keys():
                    # The conditions have different keys but neither is a subset of the other,
                    # this can not be simplified.
                    continue
                diff = set()
                for k in r1.conds:
                    if r1.conds[k] != r2.conds[k]:
                        diff.add(k)
                if len(diff) != 1:
                    continue
                # The conditions have the same keys and only one is different,
                # merge by dropping the condition with the different value
                new_conds = dict()
                for k, v in r1.conds.items():
                    if k in r2.conds and r2.conds[k] == v:
                        new_conds[k] = v
                new_rule = PolicyRule(new_conds, r1.effs)
                log.debug(f'Pruning rule {r1} and {r2} to {new_rule}')
                new_rules.add(new_rule)
                pruned_rules |= {r1, r2}
        new_rules |= self.rules - pruned_rules
        if new_rules != self.rules:
            self.rules = new_rules
            self.simplify_rules()

    def simplify_state_constraints(self):
        new_constraints = set()
        pruned_constraints = set()
        for sc1, sc2 in itertools.combinations(self.state_constraints, 2):
            c1 = sc1.conds
            c2 = sc2.conds
            if c1.keys() < c2.keys() and all([c2[k] == v for k, v in c1.items()]):
                # One condition is a subset of the other, keep the less restrictive one
                pruned_constraints.add(sc2)
                continue
            elif c1.keys() > c2.keys() and all([c1[k] == v for k, v in c2.items()]):
                # One condition is a subset of the other, keep the less restrictive one
                pruned_constraints.add(sc1)
                continue
            if c1.keys() ^ c2.keys():
                # The conditions have different keys but neither is a subset of the other,
                # this can not be simplified.
                continue
            diff = set()
            for k in c1:
                if c1[k] != c2[k]:
                    diff.add(k)
            if len(diff) != 1:
                continue
            # The conditions have the same keys and only one is different,
            # merge by dropping the condition with the different value
            new_conds = dict()
            for k, v in c1.items():
                if k in c2 and c2[k] == v:
                    new_conds[k] = v
            new_constraints.add(StateConstraint(new_conds))
        updated_state_constraints = self.state_constraints - pruned_constraints | new_constraints
        if updated_state_constraints != self.state_constraints:
            self.state_constraints = updated_state_constraints
            self.simplify_state_constraints()


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


def generate_policy(solution, policy_type=PolicyType.EXACT):
    if policy_type == PolicyType.DATALOG:
        return generate_datalog_policy(solution)
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
