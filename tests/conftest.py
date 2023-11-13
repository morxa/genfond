import pytest
import os.path
import pddl
from pddl.action import Action
from pddl.core import Domain, Problem
from pddl.requirements import Requirements
from pddl.logic import Predicate, constants, variables


@pytest.fixture
def simple_blocks():
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
    domain = Domain('simple-blocks',
                    requirements=requirements,
                    predicates=[on, holding],
                    actions=[pick, put])
    a, b = constants('a b')
    problem = Problem('p1',
                      domain=domain,
                      requirements=requirements,
                      objects=[a, b],
                      init=[on(a, b)],
                      goal=on(b, a))
    return domain, problem


@pytest.fixture
def fond_blocks():
    blocksworld_path = os.path.join(os.path.dirname(__file__), 'fixtures',
                                    'pddl_files', 'blocksworld_fond')
    domain = pddl.parse_domain(os.path.join(blocksworld_path, 'domain.pddl'))
    problem = pddl.parse_problem(os.path.join(blocksworld_path, 'p01.pddl'))
    return domain, problem
