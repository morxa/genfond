import os.path
import pddl
from pddl.action import Action
from pddl.core import Domain, Problem
from pddl.logic import Predicate, constants, variables
from pddl.requirements import Requirements
from pddl.logic.base import And, OneOf
from pddl.logic.predicates import EqualTo
from pddl.logic.effects import AndEffect, When

from genfond.ground import ground


def test_ground_simple():
    requirements = [Requirements.STRIPS]
    x, y = variables('x y')
    on = Predicate('on', x, y)
    holding = Predicate('holding', x)
    pick = Action('pick',
                  parameters=[x, y],
                  precondition=on(x, y),
                  effect=holding(x) & ~on(x, y))
    put = Action('put',
                 parameters=[x, y],
                 precondition=holding(x),
                 effect=on(x, y) & ~holding(x))
    domain = Domain('typed-blocks',
                    requirements=requirements,
                    predicates=[on, holding],
                    actions=[pick, put])
    assert domain
    a, b = constants('a b')
    problem = Problem('p1',
                      domain=domain,
                      requirements=requirements,
                      objects=[a, b],
                      init=[on(a, b)],
                      goal=[on(b, a)])
    assert problem
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
    ab_pick = [
        action for action in ground_actions
        if action.name == 'pick' and action.parameters == (a, b)
    ][0]
    assert ab_pick.precondition == on(a, b)
    assert ab_pick.effect == holding(a) & ~on(a, b)


def test_ground_blocksworld():
    blocksworld_path = os.path.join(os.path.dirname(__file__), 'fixtures',
                                    'pddl_files', 'blocksworld_fond')
    domain = pddl.parse_domain(os.path.join(blocksworld_path, 'domain.pddl'))
    problem = pddl.parse_problem(os.path.join(blocksworld_path, 'p01.pddl'))
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
    abc_puton = [
        action for action in ground_actions if action.parameters == (a, b, c)
    ][0]
    assert abc_puton.precondition == And(on(a, c), clear(a), clear(b),
                                         ~EqualTo(b, c), ~EqualTo(a, c),
                                         ~EqualTo(a, b), ~EqualTo(a, table))
    expected_effect = OneOf(
        AndEffect(on(a, b), ~on(a, c), When(~EqualTo(c, table), clear(c)),
                  When(~EqualTo(b, table), ~clear(b))),
        AndEffect(on(a, table), When(~EqualTo(c, table), ~on(a, c) & clear(c)),
                  When(~EqualTo(b, table), ~clear(b))))
    assert abc_puton.effect == expected_effect
