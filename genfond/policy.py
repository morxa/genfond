import itertools
from enum import Enum
import logging

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


class Policy:

    def __init__(self, features, rules):
        self.features = frozenset(features)
        self.rules = frozenset(rules)

    def __repr__(self):
        return '\n'.join({repr(rule) for rule in self.rules})

    def simplify(self):
        new_rules = set()
        pruned_rules = set()
        for r1, r2 in itertools.combinations(self.rules, 2):
            if r1.effs == r2.effs:
                if r1.conds.keys() < r2.conds.keys():
                    # One condition is a subset of the other, keep the less restrictive one
                    pruned_rules.add(r2)
                    continue
                elif r1.conds.keys() > r2.conds.keys():
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
            self.simplify()


def generate_policy(solution):
    rules = set()
    features = solution['selected']
    for instance, state in solution['repr']:
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
        effs = dict()
        for i, s1, s2, f, v in solution['good_trans_delta']:
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
        if effs:
            log.debug(f'Effects: {effs}')
            rule = PolicyRule(conds, frozenset({frozenset(e) for e in effs.values()}))
            log.debug(f'Adding rule {rule}')
            rules.add(rule)
    policy = Policy(features, rules)
    before_pruning = len(policy.rules)
    policy.simplify()
    log.info(f'Pruned {before_pruning - len(policy.rules)} rules')
    return policy
