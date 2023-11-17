from enum import Enum


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
        s_conds = set()
        for feature, val in self.conds.items():
            if val == Cond.TRUE:
                s_conds.add(f'{feature}')
            elif val == Cond.FALSE:
                s_conds.add(f'¬{feature}')
            elif val == Cond.POSITIVE:
                s_conds.add(f'{feature} > 0')
            elif val == Cond.ZERO:
                s_conds.add(f'{feature} = 0')
            else:
                raise ValueError(f'Unknown feature type {f}')
        s_effs = set()
        for eff in self.effs:
            s_eff = set()
            for feature, val in eff:
                if val == Effect.SET:
                    s_eff.add(f'{feature}')
                elif val == Effect.UNSET:
                    s_eff.add(f'¬{feature}')
                elif val == Effect.INCREASE:
                    s_eff.add(f'↑{feature}')
                elif val == Effect.DECREASE:
                    s_eff.add(f'↓{feature}')
                else:
                    raise ValueError(f'Unknown feature type {f}')
            s_effs.add(' ∧ '.join(s_eff))
        return f'{{ {" ∧ ".join(s_conds)} }}  ⇒  {{ {" ; ".join(s_effs)} }}'

    def __hash__(self):
        return hash(repr(self))


class Policy:

    def __init__(self, features, rules):
        self.features = frozenset(features)
        self.rules = frozenset(rules)

    def __repr__(self):
        return '\n'.join({repr(rule) for rule in self.rules})


def generate_policy(solution):
    rules = set()
    features = solution['selected']
    for instance, state in solution['repr']:
        conds = dict()
        for i, s, f, v in solution['bool_eval']:
            if i == instance and s == state:
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
                eff = effs.setdefault(s2, set())
                if f.startswith('b_'):
                    if v == 1:
                        eff.add((f, Effect.SET))
                    elif v == -1:
                        eff.add((f, Effect.UNSET))
                    elif v == 0:
                        continue
                    else:
                        raise ValueError(f'Unknown value {v}')
                elif f.startswith('n_'):
                    if v == 1:
                        eff.add((f, Effect.INCREASE))
                    elif v == -1:
                        eff.add((f, Effect.DECREASE))
                    elif v == 0:
                        continue
                    else:
                        raise ValueError(f'Unknown value {v}')
                else:
                    raise ValueError(f'Unknown value {v}')
        if effs:
            rules.add(PolicyRule(conds, frozenset({frozenset(e) for e in effs.values()})))
    return Policy(features, rules)
