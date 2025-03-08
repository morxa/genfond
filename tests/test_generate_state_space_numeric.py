from helpers import get_action
from pddl.logic import Constant
from pddl.logic.functions import NumericFunction

from genfond.ground import ground
from genfond.state_space_generator import (apply_action_effects, eval_function_term, generate_state_space)


def test_generate_state_space_childsnack(childsnack):
    domain, problem = childsnack
    actions = ground(domain, problem)
    print(", ".join(f'{action.name}({",".join([str(p) for p in action.parameters])})' for action in actions))
    state_space = generate_state_space(domain, problem)
    assert state_space.root.state == problem.init
    ngf = Constant('is_not_gluten_free', 'gluten')
    tray1 = Constant('tray1', 'tray')
    make_sandwich = get_action(actions, 'make_sandwich', (ngf, ngf))
    put_on_tray = get_action(actions, 'put_on_tray', (tray1, ngf))
    assert make_sandwich in state_space.root.children
    succ = next(iter(state_space.root.children[make_sandwich]))
    assert eval_function_term(NumericFunction('at_kitchen_sandwich', ngf), succ.state) == 1
    assert eval_function_term(NumericFunction('at_kitchen_bread', ngf), succ.state) == 3
    assert eval_function_term(NumericFunction('at_kitchen_content', ngf), succ.state) == 3
    assert make_sandwich in succ.children
    succ = next(iter(succ.children[make_sandwich]))
    assert eval_function_term(NumericFunction('at_kitchen_sandwich', ngf), succ.state) == 2
    assert eval_function_term(NumericFunction('at_kitchen_bread', ngf), succ.state) == 2
    assert eval_function_term(NumericFunction('at_kitchen_content', ngf), succ.state) == 2
    assert put_on_tray in succ.children
    succ = next(iter(succ.children[put_on_tray]))
    assert eval_function_term(NumericFunction('at_kitchen_sandwich', ngf), succ.state) == 1
    assert eval_function_term(NumericFunction('ontray', tray1, ngf), succ.state) == 1
