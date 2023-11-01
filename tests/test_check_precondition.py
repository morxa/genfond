from genfond.ground import ground
from genfond.state_space_generator import check_formula
from pddl.logic import constants


def test_precondition_check_simple_blocks(simple_blocks):
    domain, problem = simple_blocks
    state = problem.init
    a, b = constants('a b')
    grounded_actions = ground(domain, problem)
    ab_pick = [
        action for action in grounded_actions
        if action.name == 'pick' and action.parameters == (a, b)
    ][0]
    assert check_formula(state, ab_pick.precondition)
    ba_pick = [
        action for action in grounded_actions
        if action.name == 'pick' and action.parameters == (b, a)
    ][0]
    assert not check_formula(state, ba_pick.precondition)
    ab_put = [
        action for action in grounded_actions
        if action.name == 'put' and action.parameters == (a, b)
    ][0]
    assert not check_formula(state, ab_put.precondition)


def test_precondition_check_fond_blocks(fond_blocks):
    domain, problem = fond_blocks
    state = problem.init
    grounded_actions = ground(domain, problem)
    a, b, c, table = constants('A B C Table')
    cta_puton = [
        action for action in grounded_actions
        if action.name == 'puton' and action.parameters == (c, table, a)
    ][0]
    assert check_formula(state, cta_puton.precondition)
    bat_puton = [
        action for action in grounded_actions
        if action.name == 'puton' and action.parameters == (b, a, table)
    ][0]
    assert not check_formula(state, bat_puton.precondition)
