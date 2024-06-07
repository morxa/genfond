import pytest
import os.path
import pddl
from pddl.action import Action
from pddl.core import Domain, Problem
from pddl.requirements import Requirements
from pddl.logic import Predicate, constants, variables
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')


@pytest.fixture
def simple_blocks():
    requirements = [Requirements.STRIPS]
    x, y = variables('x y')
    on = Predicate('on', x, y)
    holding = Predicate('holding', x)
    pick = Action('pick', parameters=[x, y], precondition=on(x, y), effect=holding(x) & ~on(x, y))
    put = Action('put', parameters=[x, y], precondition=holding(x), effect=on(x, y) & ~holding(x))
    domain = Domain('simple-blocks', requirements=requirements, predicates=[on, holding], actions=[pick, put])
    a, b, c = constants('a b c')
    problem = Problem('p1',
                      domain=domain,
                      requirements=requirements,
                      objects=[a, b, c],
                      init=[on(a, b)],
                      goal=on(a, c))
    return domain, problem


@pytest.fixture
def fond_blocks():
    blocksworld_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'pddl_files', 'blocksworld_fond')
    domain = pddl.parse_domain(os.path.join(blocksworld_path, 'domain.pddl'))
    problem = pddl.parse_problem(os.path.join(blocksworld_path, 'p01.pddl'))
    return domain, problem


@pytest.fixture
def typed_blocks():
    typed_blocks_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'pddl_files', 'typed_blocks')
    domain = pddl.parse_domain(os.path.join(typed_blocks_path, 'domain.pddl'))
    problem = pddl.parse_problem(os.path.join(typed_blocks_path, 'p01.pddl'))
    return domain, problem


@pytest.fixture
def blocks_clear():
    blocksclear_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'pddl_files', 'blocks-clear')
    domain = pddl.parse_domain(os.path.join(blocksclear_path, 'domain.pddl'))
    problem = pddl.parse_problem(os.path.join(blocksclear_path, 'p002-1.pddl'))
    return domain, problem


@pytest.fixture
def program_with_nontriv_equiv():
    return """
        feature(n_f).
        feature_complexity(n_f, 1).
        feature(b_g).
        feature_complexity(b_g, 2).
        feature(b_h).
        feature_complexity(b_h, 3).

        state(0, 0).
        alive(0, 0).
        eval(0, 0, n_f, 0).
        eval(0, 0, b_g, 0).
        eval(0, 0, b_h, 0).

        state(0, 1).
        alive(0, 1).
        eval(0, 1, n_f, 1).
        eval(0, 1, b_g, 0).
        eval(0, 1, b_h, 0).

        state(0, 2).
        alive(0, 2).
        eval(0, 2, n_f, 1).
        eval(0, 2, b_g, 0).
        eval(0, 2, b_h, 1).

        state(0, g1).
        alive(0, g1).
        eval(0, g1, n_f, 1).
        eval(0, g1, b_g, 1).
        eval(0, g1, b_h, 0).
        goal(0, g1).

        state(0, g2).
        alive(0, g2).
        eval(0, g2, n_f, 1).
        eval(0, g2, b_g, 1).
        eval(0, g2, b_h, 1).
        goal(0, g2).


        trans(0, 0, a, 0).
        trans(0, 0, a, 1).
        trans(0, 0, a, 2).
        trans(0, 1, b, g1).
        trans(0, 1, b, 0).
        trans(0, 2, b, g2).
        trans(0, 2, b, 0).
    """


@pytest.fixture
def simple_program():
    return """
        feature(n_f).
        feature_complexity(n_f, 1).
        feature(b_g).
        feature_complexity(b_g, 2).
        feature(b_h).
        feature_complexity(b_h, 3).

        state(0, 0).
        alive(0, 0).
        eval(0, 0, n_f, 4).
        eval(0, 0, b_g, 0).
        eval(0, 0, b_h, 0).

        state(0, 1).
        alive(0, 1).
        eval(0, 1, n_f, 3).
        eval(0, 1, b_g, 0).
        eval(0, 1, b_h, 0).

        state(0, 2).
        eval(0, 2, n_f, 2).
        eval(0, 2, b_g, 1).
        eval(0, 2, b_h, 0).

        state(0, 3).
        alive(0, 3).
        eval(0, 3, n_f, 1).
        eval(0, 3, b_g, 0).
        eval(0, 3, b_h, 0).

        state(0, 4).
        alive(0, 4).
        eval(0, 4, n_f, 1).
        eval(0, 4, b_g, 0).
        eval(0, 4, b_h, 1).
        goal(0, 4).

        trans(0, 0, a, 1).
        trans(0, 1, a, 2).
        trans(0, 1, b, 3).
        trans(0, 3, a, 4).
    """
