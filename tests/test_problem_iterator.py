from dataclasses import dataclass

import pytest
from pddl.core import Plan
from pddl.logic import Constant, Predicate, constants

from genfond.problem_iterator import (
    MAX_COST,
    LastStep,
    OneShotProblemIterator,
    ProblemIterator,
    Result,
)


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


def frontier_config(**overrides):
    config = {
        "min_complexity": 2,
        "max_complexity": 7,
        "use_selected_states": False,
        "use_unrestricted_features": False,
        "unselect_problems": False,
        "min_number_of_plans": 1,
        "max_frontier_expansions": 20,
    }
    config.update(overrides)
    return config


def a_plan(*names):
    return Plan([(n, [Constant("x")]) for n in names])


def started_iterator(config):
    a, b = constants("a b")
    problems = [
        DummyProblem("p1", frozenset({Predicate("at", a)})),
        DummyProblem("p2", frozenset({Predicate("at", b)})),
    ]
    iterator = iter(ProblemIterator(problems, config))
    return iterator, next(iterator)


def test_one_shot_problem_iterator_yields_solve_step_keys():
    # solve_step takes example_plans; returning selected_states used to raise TypeError.
    config = {
        "min_complexity": 2,
        "max_complexity": 7,
        "use_selected_states": False,
        "use_unrestricted_features": True,
    }
    a, _ = constants("a b")
    problems = [DummyProblem("p1", frozenset({Predicate("at", a)}))]
    iteration = next(iter(OneShotProblemIterator(problems, config)))
    assert "example_plans" in iteration
    assert "dead_states" in iteration
    assert "selected_states" not in iteration


def test_frontier_result_does_not_tighten_max_cost():
    iterator, first = started_iterator(frontier_config())
    # A frontier model's feature cost is artificially low, so it must not become the budget
    # for the next round, and it must not mark the active problems as solved.
    iterator.set_last_result(Result.FRONTIER)
    assert iterator.max_cost == MAX_COST
    assert iterator.active_problems_solved is False

    iterator.record_frontier_expansion({"p1": [a_plan("pick")]}, {})
    second = next(iterator)
    assert iterator.last_step == LastStep.EXPAND_FRONTIER
    # The same configuration is retried, only the plan set grew.
    assert second["active_problems"] == first["active_problems"]
    assert second["complexity"] == first["complexity"]
    assert second["max_cost"] == first["max_cost"]
    assert second["example_plans"]["p1"] == [a_plan("pick")]


def test_frontier_retry_needs_progress():
    iterator, _ = started_iterator(frontier_config())
    iterator.set_last_result(Result.FRONTIER)
    # Neither a new plan nor a new dead end: retrying would repeat the identical round, so
    # the normal escalation ladder must take over instead of looping forever.
    iterator.record_frontier_expansion({}, {})
    assert iterator.frontier_progress is False
    next(iterator)
    assert iterator.last_step != LastStep.EXPAND_FRONTIER


def test_new_dead_states_count_as_progress():
    iterator, _ = started_iterator(frontier_config())
    iterator.set_last_result(Result.FRONTIER)
    iterator.record_frontier_expansion({}, {"p1": {frozenset({"s"})}})
    assert iterator.frontier_progress is True
    # Re-reporting the same dead end is not progress.
    iterator.set_last_result(Result.FRONTIER)
    iterator.record_frontier_expansion({}, {"p1": {frozenset({"s"})}})
    assert iterator.frontier_progress is False


def test_max_frontier_expansions_stops_the_retry_loop():
    iterator, _ = started_iterator(frontier_config(max_frontier_expansions=2))
    for i in range(3):
        iterator.set_last_result(Result.FRONTIER)
        # Distinct each round, so the retry is stopped by the budget and not by deduplication.
        iterator.record_frontier_expansion({"p1": [a_plan(f"pick{i}")]}, {})
        next(iterator)
    assert iterator.frontier_expansions == 3
    iterator.set_last_result(Result.FRONTIER)
    iterator.record_frontier_expansion({"p1": [a_plan("another")]}, {})
    next(iterator)
    assert iterator.last_step != LastStep.EXPAND_FRONTIER


def test_duplicate_frontier_plans_are_not_progress():
    iterator, _ = started_iterator(frontier_config())
    iterator.set_last_result(Result.FRONTIER)
    iterator.record_frontier_expansion({"p1": [a_plan("pick", "put")]}, {})
    assert iterator.frontier_progress is True
    assert len(iterator.active_plans["p1"]) == 1

    # The planner is deterministic: a frontier state the plan failed to expand yields the
    # identical plan next round. Re-adding it would spin the retry loop.
    iterator.set_last_result(Result.FRONTIER)
    iterator.record_frontier_expansion({"p1": [a_plan("pick", "put")]}, {})
    assert iterator.frontier_progress is False
    assert len(iterator.active_plans["p1"]) == 1

    # A genuinely different plan still counts.
    iterator.set_last_result(Result.FRONTIER)
    iterator.record_frontier_expansion({"p1": [a_plan("pick", "drop")]}, {})
    assert iterator.frontier_progress is True
    assert len(iterator.active_plans["p1"]) == 2
