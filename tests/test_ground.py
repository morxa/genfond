from helpers import get_action
from pddl.action import Action
from pddl.logic import Predicate, constants, variables
from pddl.logic.base import And, OneOf
from pddl.logic.effects import When
from pddl.logic.predicates import EqualTo
from pddl.logic.terms import Constant

from genfond.ground import ground, ground_domain_predicates


def test_ground_simple(simple_blocks):
    domain, problem = simple_blocks
    a, b, c = constants("a b c")
    x, y = variables("x y")
    on = Predicate("on", x, y)
    holding = Predicate("holding", x)
    ground_actions = ground(domain, problem)
    assert len(ground_actions) == 18
    assert all(isinstance(action, Action) for action in ground_actions)
    assert all(action.name in ("pick", "put") for action in ground_actions)
    assert all(len(action.parameters) == 2 for action in ground_actions)
    assert any(action.parameters == (a, a) for action in ground_actions)
    assert all(all(p in (a, b, c) for p in action.parameters) for action in ground_actions)
    assert {action.precondition for action in ground_actions} == {
        on(a, a),
        on(a, b),
        on(a, c),
        on(b, a),
        on(b, b),
        on(b, c),
        on(c, a),
        on(c, b),
        on(c, c),
        holding(a),
        holding(b),
        holding(c),
    }
    ab_pick = get_action(ground_actions, "pick", (a, b))
    assert ab_pick.precondition == on(a, b)
    assert ab_pick.effect == holding(a) & ~on(a, b)


def test_ground_blocksworld(fond_blocks):
    domain, problem = fond_blocks
    ground_actions = ground(domain, problem)
    x, y = variables("x y")
    on = Predicate("on", x, y)
    clear = Predicate("clear", x)
    a, b, c, table = constants("A B C Table")
    assert len(ground_actions) == 4**3
    assert all(isinstance(action, Action) for action in ground_actions)
    assert all(action.name == "puton" for action in ground_actions)
    assert all(len(action.parameters) == 3 for action in ground_actions)
    assert any(action.parameters == (a, a, a) for action in ground_actions)
    assert any(action.parameters == (a, b, table) for action in ground_actions)
    assert any(action.parameters == (a, b, c) for action in ground_actions)
    abc_puton = get_action(ground_actions, "puton", (a, b, c))
    assert abc_puton.precondition == And(
        on(a, c), clear(a), clear(b), ~EqualTo(b, c), ~EqualTo(a, c), ~EqualTo(a, b), ~EqualTo(a, table)
    )
    expected_effect = OneOf(
        And(on(a, b), ~on(a, c), When(~EqualTo(c, table), clear(c)), When(~EqualTo(b, table), ~clear(b))),
        And(on(a, table), When(~EqualTo(c, table), ~on(a, c) & clear(c))),
    )
    assert abc_puton.effect == expected_effect


def test_ground_predicates(simple_blocks):
    domain, problem = simple_blocks
    ground_predicates = ground_domain_predicates(domain, problem)
    assert len(ground_predicates) == 12
    a, b, c = constants("a b c")
    assert ground_predicates == {
        Predicate("on", a, a),
        Predicate("on", a, b),
        Predicate("on", a, c),
        Predicate("on", b, a),
        Predicate("on", b, b),
        Predicate("on", b, c),
        Predicate("on", c, a),
        Predicate("on", c, b),
        Predicate("on", c, c),
        Predicate("holding", a),
        Predicate("holding", b),
        Predicate("holding", c),
    }


def test_ground_typed(typed_blocks):
    domain, problem = typed_blocks
    a, b = constants("a b", "block")
    table = Constant("table", "obj")
    ground_actions = ground(domain, problem)
    assert ground_actions
    print(ground_actions)
    for action in ground_actions:
        assert action.parameters != (table, a)
    assert any(action.name == "put" and action.parameters == (a, b) for action in ground_actions)


def test_ground_typed_predicates(typed_blocks):
    domain, problem = typed_blocks
    a, b = constants("a b", "block")
    table = Constant("table", "obj")
    ground_predicates = ground_domain_predicates(domain, problem)
    assert ground_predicates == {
        Predicate("on", a, a),
        Predicate("on", a, b),
        Predicate("on", b, a),
        Predicate("on", b, b),
        Predicate("on", a, table),
        Predicate("on", b, table),
        Predicate("holding", a),
        Predicate("holding", b),
    }
