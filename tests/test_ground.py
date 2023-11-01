from pddl.action import Action
from pddl.logic import Predicate, constants, variables
from pddl.logic.base import And, OneOf
from pddl.logic.predicates import EqualTo
from pddl.logic.effects import AndEffect, When

from genfond.ground import ground

from helpers import get_action


def test_ground_simple(simple_blocks):
    domain, problem = simple_blocks
    a, b = constants('a b')
    x, y = variables('x y')
    on = Predicate('on', x, y)
    holding = Predicate('holding', x)
    ground_actions = ground(domain, problem)
    assert len(ground_actions) == 8
    assert all(isinstance(action, Action) for action in ground_actions)
    assert all(action.name in ('pick', 'put') for action in ground_actions)
    assert all(len(action.parameters) == 2 for action in ground_actions)
    assert any(action.parameters == (a, a) for action in ground_actions)
    assert all(
        all(p in (a, b) for p in action.parameters)
        for action in ground_actions)
    assert all(action.precondition in (on(a, a), on(a, b), on(b, a), on(b, b),
                                       holding(a), holding(b))
               for action in ground_actions)
    ab_pick = get_action(ground_actions, 'pick', (a, b))
    assert ab_pick.precondition == on(a, b)
    assert ab_pick.effect == holding(a) & ~on(a, b)


def test_ground_blocksworld(fond_blocks):
    domain, problem = fond_blocks
    ground_actions = ground(domain, problem)
    x, y = variables('x y')
    on = Predicate('on', x, y)
    clear = Predicate('clear', x)
    a, b, c, table = constants('A B C Table')
    assert len(ground_actions) == 4**3
    assert all(isinstance(action, Action) for action in ground_actions)
    assert all(action.name == 'puton' for action in ground_actions)
    assert all(len(action.parameters) == 3 for action in ground_actions)
    assert any(action.parameters == (a, a, a) for action in ground_actions)
    assert any(action.parameters == (a, b, table) for action in ground_actions)
    assert any(action.parameters == (a, b, c) for action in ground_actions)
    abc_puton = get_action(ground_actions, 'puton', (a, b, c))
    assert abc_puton.precondition == And(on(a, c), clear(a), clear(b),
                                         ~EqualTo(b, c), ~EqualTo(a, c),
                                         ~EqualTo(a, b), ~EqualTo(a, table))
    expected_effect = OneOf(
        AndEffect(on(a, b), ~on(a, c), When(~EqualTo(c, table), clear(c)),
                  When(~EqualTo(b, table), ~clear(b))),
        AndEffect(on(a, table), When(~EqualTo(c, table), ~on(a, c) & clear(c)),
                  When(~EqualTo(b, table), ~clear(b))))
    assert abc_puton.effect == expected_effect
