from pddl.logic.base import And, Not, OneOf
from pddl.logic import Predicate
from pddl.logic.predicates import EqualTo
from collections.abc import Collection
from pddl.logic.effects import AndEffect, When
from .ground import ground
from enum import Enum


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


class StateSpaceNode:

    def __init__(self, state, id):
        self.state = state
        self.id = id
        self.children = dict()
        self.alive = Alive.UNKNOWN
        self.parents = set()

    def __str__(self):
        return ','.join([f'{p.name}({",".join([str(p) for p in p.terms])})' for p in self.state])

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
                        if check_formula(new_node.state, problem.goal):
                            new_node.alive = Alive.ALIVE
                        queue.append(new_node)
        self.compute_distances()
        compute_alive(self.nodes.values())
        assert all(node.alive != Alive.UNKNOWN for node in self.nodes.values())
        assert self.root.alive == Alive.ALIVE, 'Problem {} is unsolvable'.format(problem.name)

    def compute_distances(self):
        parents = {node: set() for node in self.nodes.values()}
        for node in self.nodes.values():
            for action, children in node.children.items():
                for child in children:
                    parents[child].add(node)
        for node in self.nodes.values():
            node.distance = len(self.nodes)
        goal_nodes = [node for node in self.nodes.values() if check_formula(node.state, self.problem.goal)]
        for node in goal_nodes:
            node.distance = 0
        queue = goal_nodes
        while queue:
            node = queue.pop()
            for parent in parents[node]:
                if parent.distance is None or parent.distance > node.distance + 1:
                    parent.distance = node.distance + 1
                    queue.append(parent)
        assert all(node.distance is not None for node in self.nodes.values())

    def add_node(self, state, parent_state, action):
        parent = self.nodes[parent_state]
        try:
            parent.add_child(action, self.nodes[state])
            self.nodes[state].parents.add(parent)
            return None
        except KeyError:
            node = StateSpaceNode(state, self.next_id)
            self.next_id += 1
            self.nodes[state] = node
            parent.add_child(action, node)
            node.parents.add(parent)
            return node


def generate_state_space(domain, problem):
    return StateSpaceGraph(domain, problem)


def can_reach_goal(node, goal_nodes, seen=None):
    if not seen:
        seen = []
    if node in goal_nodes:
        return True
    if node.alive == Alive.DEAD:
        return False
    if node.alive == Alive.ALIVE:
        return True
    if node in seen:
        return False
    seen.append(node)
    for children in node.children.values():
        if any(child.alive == Alive.DEAD for child in children):
            continue
        for child in children:
            if can_reach_goal(child, goal_nodes, seen):
                return True
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
    goal_nodes = [node for node in nodes if node.alive == Alive.ALIVE]
    changed = False
    while queue:
        node = queue.pop()
        if not can_reach_goal(node, goal_nodes):
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
