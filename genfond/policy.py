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


class Policy:

    def __init__(self, solution):
        self.rules = set()
        for instance, state in solution['repr']:
            conds = set()
            for i, s, f, v in solution['bool_eval']:
                if i == instance and s == state:
                    if f.startswith('b_'):
                        if v == 1:
                            conds.add((f, Cond.TRUE))
                        elif v == 0:
                            conds.add((f, Cond.FALSE))
                        else:
                            raise ValueError(f'Unknown value {v}')
                    elif f.startswith('n_'):
                        if v == 1:
                            conds.add((f, Cond.POSITIVE))
                        elif v == 0:
                            conds.add((f, Cond.ZERO))
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
                self.rules.add((frozenset(conds), frozenset({frozenset(e) for e in effs.values()})))

    def __repr__(self):
        s_rules = []
        for conds, effs in self.rules:
            s_conds = []
            for feature, val in conds:
                if val == Cond.TRUE:
                    s_conds.append(f'{feature}')
                elif val == Cond.FALSE:
                    s_conds.append(f'¬{feature}')
                elif val == Cond.POSITIVE:
                    s_conds.append(f'{feature} > 0')
                elif val == Cond.ZERO:
                    s_conds.append(f'{feature} = 0')
                else:
                    raise ValueError(f'Unknown feature type {f}')
            s_effs = []
            for eff in effs:
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
                        raise ValueError(f'Unknown feature type {f}')
                s_effs.append(' ∧ '.join(s_eff))
            s_rules.append(f'{{ {" ∧ ".join(s_conds)} }}  ⇒  {{ {" ; ".join(s_effs)} }}')
        return '\n'.join(s_rules)
