from pddl.logic.base import And, Not
from pddl.logic import Predicate
from pddl.logic.predicates import EqualTo


def check_formula(state, formula):
    if isinstance(formula, And):
        return all(
            check_formula(state, subformula)
            for subformula in formula.operands)
    elif isinstance(formula, Not):
        return not check_formula(state, formula.argument)
    elif isinstance(formula, Predicate):
        return formula in state
    elif isinstance(formula, EqualTo):
        return formula.left == formula.right
    else:
        raise ValueError('Unknown formula type: {}'.format(type(formula)))
