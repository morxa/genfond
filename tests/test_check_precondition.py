from helpers import get_action
from pddl.logic import constants

from genfond.ground import ground
from genfond.state_space_generator import check_formula


def test_precondition_check_simple_blocks(simple_blocks):
    domain, problem = simple_blocks
    state = problem.init
    a, b = constants("a b")
    grounded_actions = ground(domain, problem)
    ab_pick = get_action(grounded_actions, "pick", (a, b))
    assert check_formula(state, ab_pick.precondition)
    aa_pick = get_action(grounded_actions, "pick", (a, a))
    assert not check_formula(state, aa_pick.precondition)
    ba_pick = get_action(grounded_actions, "pick", (b, a))
    assert not check_formula(state, ba_pick.precondition)
    ab_put = get_action(grounded_actions, "put", (a, b))
    assert not check_formula(state, ab_put.precondition)


def test_precondition_check_fond_blocks(fond_blocks):
    domain, problem = fond_blocks
    state = problem.init
    grounded_actions = ground(domain, problem)
    a, b, c, table = constants("A B C Table")
    cta_puton = get_action(grounded_actions, "puton", (c, table, a))
    assert check_formula(state, cta_puton.precondition)
    bat_puton = get_action(grounded_actions, "puton", (b, a, table))
    assert not check_formula(state, bat_puton.precondition)
