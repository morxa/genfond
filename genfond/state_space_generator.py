from pddl.logic.base import And, Not, OneOf
from pddl.logic import Predicate
from pddl.logic.predicates import EqualTo
from collections.abc import Collection, Set
from pddl.logic.effects import AndEffect, When


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


def apply_action_effects(state, action):
    if isinstance(action.effect, OneOf):
        return {
            apply_effects(state, effect)
            for effect in action.effect.operands
        }
    else:
        return {apply_effects(state, action.effect)}


def apply_effects(state, effects):
    if isinstance(effects, Collection):
        for effect in effects:
            state = apply_effects(state, effect)
        return state
    elif isinstance(effects, And):
        for effect in effects.operands:
            state = apply_effects(state, effect)
        return state
    elif isinstance(effects, AndEffect):
        for effect in effects.operands:
            state = apply_effects(state, effect)
        return state
    elif isinstance(effects, Predicate):
        return frozenset(state | {effects})
    elif isinstance(effects, Not):
        return frozenset([f for f in state if f != effects.argument])
    elif isinstance(effects, When):
        if check_formula(state, effects.condition):
            return apply_effects(state, effects.effect)
        else:
            return state
    else:
        raise ValueError('Unknown effect type: {}'.format(type(effects)))
