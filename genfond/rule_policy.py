import itertools
import logging
from enum import Enum

from .policy import PolicyType

log = logging.getLogger("genfond.policy.rule")


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


def cond_to_str(feature, val, param_sub=None):
    if param_sub:
        feature = feature.replace("c_primitive(aparam0,0)", param_sub)
    if val == Cond.TRUE:
        return f"{feature}"
    elif val == Cond.FALSE:
        return f"¬{feature}"
    elif val == Cond.POSITIVE:
        return f"{feature} > 0"
    elif val == Cond.ZERO:
        return f"{feature} = 0"
    else:
        raise ValueError(f"Unexpected value {val} for feature {feature}")


def eff_to_str(feature, val):
    if val == Effect.SET:
        return f"{feature}"
    elif val == Effect.UNSET:
        return f"¬{feature}"
    elif val == Effect.INCREASE:
        return f"↑{feature}"
    elif val == Effect.DECREASE:
        return f"↓{feature}"
    else:
        raise ValueError(f"Unexpected value {val} for feature {feature}")


class PolicyRule:

    def __init__(self, conds, effs):
        self.conds = conds
        self.effs = frozenset({frozenset(eff) for eff in effs})

    def __eq__(self, other):
        return self.conds == other.conds and self.effs == other.effs

    def __repr__(self):
        s_conds = []
        for feature, val in self.conds.items():
            s_conds.append(cond_to_str(feature, val))
        s_effs = []
        for eff in self.effs:
            s_eff = [eff_to_str(feature, val) for feature, val in eff]
            s_effs.append(" ∧ ".join(s_eff))
        return f'{{ {" ∧ ".join(sorted(s_conds))} }}  ⇒  {{ {" ; ".join(sorted(s_effs))} }}'

    def __hash__(self):
        return hash(repr(self))


class StateConstraint:

    def __init__(self, conds):
        self.conds = conds

    def __eq__(self, other):
        return self.conds == other.conds

    def __repr__(self):
        s_conds = [cond_to_str(feature, val) for feature, val in self.conds.items()]
        return f'{{ {" ∧ ".join(sorted(s_conds))} }}'

    def __hash__(self):
        return hash(repr(self))


class Policy:

    def __init__(
        self,
        features,
        rules,
        cost=None,
        constraints=None,
        state_constraints=None,
        type=PolicyType.EXACT,
    ):
        self.type = type
        self.features = frozenset(features)
        self.rules = frozenset(rules)
        self.constraints = constraints or set()
        self.state_constraints = state_constraints or set()
        self.cost = cost
        assert not constraints or type == PolicyType.CONSTRAINED

    def __repr__(self):
        if self.type == PolicyType.CONSTRAINED:
            s_constraints = ""
            s_state_constraints = ""
            if self.constraints:
                s_constraints = "\nConstraints:\n" + "\n".join(
                    sorted({repr(constraint) for constraint in self.constraints})
                )
            if self.state_constraints:
                s_state_constraints = "\nState Constraints:\n" + "\n".join(
                    sorted({repr(constraint) for constraint in self.state_constraints})
                )

            return (
                "{} rules,"
                " {} transition constraints,"
                " {} state constraints with {} features: {}"
                "\nRules:\n{}{}{}"
            ).format(
                len(self.rules),
                len(self.constraints),
                len(self.state_constraints),
                len(self.features),
                ", ".join(sorted(self.features)),
                "\n".join(sorted({repr(rule) for rule in self.rules})),
                s_constraints,
                s_state_constraints,
            )
        else:
            return "{} rules with {} features: {}\n{}".format(
                len(self.rules),
                len(self.features),
                ", ".join(sorted(self.features)),
                "\n".join(sorted({repr(rule) for rule in self.rules})),
            )

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
                log.debug(f"Pruning rule {r1} and {r2} to {new_rule}")
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
