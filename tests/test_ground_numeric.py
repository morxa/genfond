from helpers import get_action
from pddl.action import Action
from pddl.logic import Constant, Variable
from pddl.logic.base import And
from pddl.logic.functions import Decrease, LesserEqualThan, NumericFunction, NumericValue
from pddl.logic.predicates import Predicate

from genfond.ground import ground


def test_ground_childsnack(childsnack):
    domain, problem = childsnack
    ground_actions = ground(domain, problem)
    # make_sandwich_no_gluten: 4, make_sandwich: 4, put_on_tray: 2, serve_sandwich_no_gluten: 4,
    # serve_sandwich: 8, move_tray: 4**2
    assert len(ground_actions) == 4 + 4 + 2 + 4 + 8 + 4**2
    assert all(isinstance(action, Action) for action in ground_actions)
    assert {action.name for action in ground_actions} == {
        "make_sandwich_no_gluten",
        "make_sandwich",
        "put_on_tray",
        "serve_sandwich_no_gluten",
        "serve_sandwich",
        "move_tray",
    }
    assert all(len(action.parameters) == 2 for action in ground_actions if action.name == "make_sandwich_no_gluten")
    assert all(len(action.parameters) == 2 for action in ground_actions if action.name == "make_sandwich")
    assert all(len(action.parameters) == 2 for action in ground_actions if action.name == "put_on_tray")
    assert all(len(action.parameters) == 2 for action in ground_actions if action.name == "serve_sandwich_no_gluten")
    assert all(len(action.parameters) == 3 for action in ground_actions if action.name == "serve_sandwich")
    assert all(len(action.parameters) == 3 for action in ground_actions if action.name == "move_tray")

    tray1 = Constant("tray1", "tray")
    is_gluten_free = Constant("is_gluten_free", "gluten")
    is_not_gluten_free = Constant("is_not_gluten_free", "gluten")
    kitchen = Constant("kitchen", "place")
    x = Variable("x", {"tray"})
    y = Variable("y", {"place"})
    at = Predicate("at", x, y)

    print([action for action in ground_actions if action.name == "serve_sandwich"])
    serve_sandwich = get_action(ground_actions, "serve_sandwich", (tray1, kitchen, is_gluten_free))
    assert serve_sandwich.precondition == And(
        at(tray1, kitchen),
        LesserEqualThan(NumericValue(1), NumericFunction("ontray", tray1, is_gluten_free)),
        LesserEqualThan(NumericValue(1), NumericFunction("hungry", kitchen, is_not_gluten_free)),
    )
    assert serve_sandwich.effect == And(
        Decrease(NumericFunction("ontray", tray1, is_gluten_free), NumericValue(1)),
        Decrease(NumericFunction("hungry", kitchen, is_not_gluten_free), NumericValue(1)),
    )
