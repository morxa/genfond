from pddl.logic.base import And, Not, OneOf
from pddl.logic import Predicate
from pddl.logic.predicates import EqualTo
from collections.abc import Collection
from pddl.logic.effects import AndEffect, When
from .ground import ground
from enum import Enum

import logging

log = logging.getLogger(__name__)


def state_to_string(state):
    return ','.join([f'{p.name}({",".join([str(p) for p in p.terms])})' for p in state])


def check_formula(state, formula):
    if isinstance(formula, And):
        return all(check_formula(state, subformula) for subformula in formula.operands)
    elif isinstance(formula, Not):
        return not check_formula(state, formula.argument)
    elif isinstance(formula, Predicate):
        return formula in state
    elif isinstance(formula, EqualTo):
        return formula.left == formula.right
    else:
        raise ValueError('Unknown formula type: {}'.format(type(formula)))


def apply_action_effects(state, action):
    return apply_effects(frozenset({state}), action.effect)


def apply_effects(states, effects):
    new_states = set()
    for state in states:
        new_states |= apply_effects_to_state(state, effects)
        assert all(isinstance(s, Collection) for s in new_states)
        assert all(all(isinstance(f, Predicate) for f in s) for s in new_states)
    return frozenset(new_states)


def apply_effects_to_state(state, effects):
    assert all(isinstance(f, Predicate) for f in state)
    if isinstance(effects, Collection):
        states = {state}
        for effect in effects:
            states = apply_effects(states, effect)
        return states
    elif isinstance(effects, And) or isinstance(effects, AndEffect):
        states = {state}
        for effect in effects.operands:
            states = apply_effects(states, effect)
        return frozenset(states)
    elif isinstance(effects, Predicate):
        return frozenset({state | {effects}})
    elif isinstance(effects, Not):
        return frozenset({frozenset([f for f in state if f != effects.argument])})
    elif isinstance(effects, When):
        if check_formula(state, effects.condition):
            return apply_effects({state}, effects.effect)
        else:
            return frozenset({state})
    elif isinstance(effects, OneOf):
        new_states = set()
        for effect in effects.operands:
            new_states |= apply_effects({state}, effect)
        return frozenset(new_states)
    else:
        raise ValueError('Unknown effect type: {}'.format(type(effects)))


class Alive(Enum):
    ALIVE = 0
    DEAD = 1
    UNKNOWN = 2
    PRUNED = 3


class StateSpaceNode:

    def __init__(self, state, id):
        self.state = state
        self.id = id
        self.children = dict()
        self.alive = Alive.UNKNOWN
        self.goal = False
        self.parents = set()

    def __str__(self):
        return state_to_string(self.state)

    def __repr__(self):
        return repr(self.state)

    def add_child(self, action, node):
        self.children.setdefault(action, set()).add(node)


class StateSpaceGraph:

    def __init__(self, domain, problem, prune=True, selected_states=None):
        self.domain = domain
        self.problem = problem
        self.next_id = 0
        queue = []
        self.nodes = dict()
        if selected_states:
            for state in selected_states:
                node = StateSpaceNode(state, self.next_id)
                self.next_id += 1
                self.nodes[state] = node
                queue.append(node)
        else:
            root_state = problem.init
            self.root = StateSpaceNode(root_state, 0)
            self.next_id = 1
            self.nodes = {root_state: self.root}
            queue = [self.root]
        grounded_actions = ground(domain, problem)
        while queue:
            node = queue.pop()
            state = node.state
            if check_formula(state, problem.goal):
                node.alive = Alive.ALIVE
                node.goal = True
            for action in grounded_actions:
                if not check_formula(state, action.precondition):
                    continue
                for succ in apply_action_effects(node.state, action):
                    new_node = self.add_node(succ, state, action)
                    if new_node:
                        if selected_states and succ not in selected_states:
                            new_node.alive = Alive.PRUNED
                        else:
                            queue.append(new_node)
        compute_alive(self.nodes.values())
        if prune:
            self.prune_nodes()
        assert all(node.alive != Alive.UNKNOWN for node in self.nodes.values())
        #assert self.root.alive == Alive.ALIVE, 'Problem {} is unsolvable'.format(problem.name)

    def add_node(self, state, parent_state, action):
        parent = self.nodes[parent_state]
        try:
            node = self.nodes[state]
            new = False
        except KeyError:
            node = StateSpaceNode(state, self.next_id)
            self.next_id += 1
            self.nodes[state] = node
            new = True

        parent.add_child(action, node)
        node.parents.add(parent)
        if new:
            return node
        else:
            return None

    def prune_nodes(self):
        pruned_dead = []
        pruned_selected = []
        for state, node in self.nodes.items():
            if node.alive == Alive.DEAD and all([parent.alive == Alive.DEAD for parent in node.parents]):
                pruned_dead.append(state)
        before = len(self.nodes)
        for state in pruned_dead + pruned_selected:
            del self.nodes[state]
        for node in self.nodes.values():
            for action, children in node.children.items():
                node.children[action] = {child for child in children if child.state in self.nodes}
        log.info(f'Pruned {len(pruned_dead)} dead '
                 f'out of {before} states in {self.problem.name}')


def generate_state_space(domain, problem, selected_states=None):
    return StateSpaceGraph(domain, problem, selected_states=selected_states)


def can_reach(node, goal_nodes):
    seen = set()
    stack = [node]
    while stack:
        current_node = stack.pop()
        if current_node in goal_nodes:
            return True
        if current_node.alive == Alive.DEAD:
            continue
        if current_node.alive == Alive.ALIVE:
            return True
        if current_node in seen:
            continue
        seen.add(current_node)
        for children in current_node.children.values():
            if all(child.alive != Alive.DEAD for child in children):
                stack.extend(children)
    return False


def find_nodes_leading_to_dead(nodes):
    queue = [node for node in nodes if node.alive == Alive.UNKNOWN]
    changed = False
    while queue:
        node = queue.pop()
        if all(any(child.alive == Alive.DEAD for child in children) for children in node.children.values()):
            node.alive = Alive.DEAD
            changed = True
            for parent in node.parents:
                if parent.alive == Alive.UNKNOWN:
                    queue.append(parent)
    return changed


def find_node_not_reaching_goal(nodes):
    queue = [node for node in nodes if node.alive == Alive.UNKNOWN]
    goal_nodes = [node for node in nodes if node.alive in [Alive.ALIVE, Alive.PRUNED]]
    changed = False
    while queue:
        node = queue.pop()
        if not can_reach(node, goal_nodes):
            node.alive = Alive.DEAD
            changed = True
            for parent in node.parents:
                if parent.alive == Alive.UNKNOWN:
                    queue.append(parent)
    return changed


def compute_alive(nodes):
    changed = True
    while changed:
        changed = find_nodes_leading_to_dead(nodes)
        changed = find_node_not_reaching_goal(nodes) or changed
    for node in nodes:
        if node.alive == Alive.UNKNOWN:
            node.alive = Alive.ALIVE
