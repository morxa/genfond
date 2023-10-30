from pddl.action import Action
from pddl.core import Domain, Problem
from pddl.logic import Predicate, constants, variables
from pddl.requirements import Requirements

from genfond.ground import ground


def test_ground():
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
    assert any(action.precondition == on(a, b) and action.effect == holding(a)
               & ~on(a, b) for action in ground_actions)
