from dataclasses import dataclass

import pytest
from pddl.logic import Predicate, constants

from genfond.problem_iterator import MAX_COST, OneShotProblemIterator


@dataclass(frozen=True)
class DummyProblem:
    name: str
    init: frozenset[str]


def test_one_shot_problem_iterator_returns_single_max_complexity_item():
    a, b = constants("a b")
    problems = [
        DummyProblem("p1", frozenset({Predicate("at", a)})),
        DummyProblem("p2", frozenset({Predicate("at", b)})),
    ]
    config = {
        "min_complexity": 2,
        "max_complexity": 7,
        "use_selected_states": False,
        "use_unrestricted_features": True,
    }
    iterator = iter(OneShotProblemIterator(problems, config))

    iteration = next(iterator)

    assert iteration["active_problems"] == problems
    assert iteration["complexity"] == config["max_complexity"]
    assert iteration["all_features"] is True
    assert iteration["max_cost"] == MAX_COST
    with pytest.raises(StopIteration):
        next(iterator)
