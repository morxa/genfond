from genfond.state_space_generator import apply_effects, apply_action_effects
from helpers import get_action
from genfond.ground import ground
from pddl.logic import constants, variables, Predicate


def test_apply_effects_simple_blocks(simple_blocks):
    domain, problem = simple_blocks
    ground_actions = ground(domain, problem)
    a, b = constants('a b')
    holding = Predicate('holding', a)
    states = apply_action_effects(problem.init, get_action(ground_actions, 'pick', (a, b)))
    assert states == frozenset([frozenset([holding(a)])])
    states = apply_action_effects(next(iter(states)), get_action(ground_actions, 'put', (a, b)))
    assert states == {problem.init}


def test_apply_effects_fond_blocks(fond_blocks):
    domain, problem = fond_blocks
    ground_actions = ground(domain, problem)
    x, y = variables('x y')
    a, b, c, table = constants('A B C Table')
    on = Predicate('on', x, y)
    clear = Predicate('clear', x)
    block = Predicate('block', x)
    blocks = set([block(a), block(b), block(c), block(table)])
    states = apply_action_effects(problem.init, get_action(ground_actions, 'puton', (c, b, a)))
    assert len(states) == 2
    assert states == {
        frozenset(blocks
                  | {on(a, table), clear(a), on(b, table),
                     on(c, b), clear(c), clear(table)}),
        frozenset(blocks
                  | {on(a, table),
                     clear(a), on(b, table),
                     clear(b), on(c, table),
                     clear(c), clear(table)})
    }


def test_apply_effects_goal_state(simple_blocks):
    domain, problem = simple_blocks
    goal_states = apply_effects(set([frozenset()]), problem.goal)
    a, b, c = constants('a b c')
    assert goal_states == {frozenset([Predicate('on', a, c)])}
