from pddl.logic.base import And, Not, OneOf
from pddl.logic import Predicate
from pddl.logic.predicates import EqualTo
from collections.abc import Collection
from pddl.logic.effects import AndEffect, When
from .ground import ground


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


class StateSpaceNode:

    def __init__(self, state, id):
        self.state = state
        self.id = id
        self.children = dict()

    def __str__(self):
        return ','.join([
            f'{p.name}({",".join([str(p) for p in p.terms])})'
            for p in self.state
        ])

    def __repr__(self):
        return repr(self.state)

    def add_child(self, action, node):
        self.children.setdefault(action, set()).add(node)


class StateSpaceGraph:

    def __init__(self, domain, problem):
        self.domain = domain
        self.problem = problem
        root_state = problem.init
        self.root = StateSpaceNode(root_state, 0)
        self.next_id = 1
        self.nodes = {root_state: self.root}
        queue = [self.root]
        grounded_actions = ground(domain, problem)
        while queue:
            node = queue.pop()
            state = node.state
            for action in grounded_actions:
                if not check_formula(state, action.precondition):
                    continue
                for succ in apply_action_effects(node.state, action):
                    new_node = self.add_node(succ, state, action)
                    if new_node:
                        queue.append(new_node)

    def add_node(self, state, parent_state, action):
        parent = self.nodes[parent_state]
        try:
            parent.add_child(action, self.nodes[state])
            return None
        except KeyError:
            node = StateSpaceNode(state, self.next_id)
            self.next_id += 1
            self.nodes[state] = node
            parent.add_child(action, node)
            return node


def generate_state_space(domain, problem):
    return StateSpaceGraph(domain, problem)
