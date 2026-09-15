from dataclasses import dataclass

import pytest
from pddl.core import Plan
from pddl.logic import Constant, Predicate, constants

from genfond.problem_iterator import (
    LastStep,
    PlanStateCoverage,
    ProblemIterator,
    Result,
    plan_key,
)
from genfond.state_space_generator import plan_visited_states


def test_plan_visited_states_follows_the_plan_deterministically(simple_blocks):
    domain, problem = simple_blocks
    a, b, c = constants("a b c")
    plan = Plan([("pick", [a, b]), ("put", [a, c])])

    visited = plan_visited_states(domain, problem, plan)

    on_ab = Predicate("on", a, b)
    holding_a = Predicate("holding", a)
    on_ac = Predicate("on", a, c)
    assert visited == {
        frozenset({on_ab}),
        frozenset({holding_a}),
        frozenset({on_ac}),
    }


def test_plan_state_coverage_deduplicates_by_state_not_action_sequence(simple_blocks):
    domain, problem = simple_blocks
    a, b, c = constants("a b c")
    coverage = PlanStateCoverage(domain, {problem.name: problem})

    first = Plan([("pick", [a, b]), ("put", [a, c])])
    assert coverage.add(problem.name, first) is True

    # A different action sequence (put back where it started, then pick again) that only ever
    # visits states the first plan already covers -- plan_key alone would not catch this since
    # the action sequences differ.
    subsumed = Plan([("pick", [a, b]), ("put", [a, b])])
    assert coverage.add(problem.name, subsumed) is False

    # A plan that does reach a new state must still be recognised as new.
    genuinely_new = Plan([("pick", [a, b]), ("put", [a, b]), ("pick", [a, b]), ("put", [a, c])])
    # Its states are the union of the two above plus nothing else, so it adds nothing new either.
    assert coverage.add(problem.name, genuinely_new) is False


@dataclass(frozen=True)
class DummyProblem:
    name: str
    init: frozenset[str]


def a_plan(*names):
    return Plan([(n, [Constant("x")]) for n in names])


def frontier_config(**overrides):
    config = {
        "min_complexity": 2,
        "max_complexity": 7,
        "use_selected_states": False,
        "use_unrestricted_features": False,
        "unselect_problems": False,
        "min_number_of_plans": 1,
        "max_frontier_expansions": 20,
        "reset_complexity_on_state_space_change": False,
        "add_problem_after_success": False,
    }
    config.update(overrides)
    return config


def iterator_with_plans(config, num_plans=10, **kwargs):
    a, b = constants("a b")
    problems = [
        DummyProblem("p1", frozenset({Predicate("at", a)})),
        DummyProblem("p2", frozenset({Predicate("at", b)})),
    ]
    plans = {p.name: iter([a_plan(f"{p.name}-{i}") for i in range(num_plans)]) for p in problems}
    iterator = iter(ProblemIterator(problems, config, plans=plans, **kwargs))
    return iterator, next(iterator)


def refute(iterator):
    iterator.set_last_result(Result.NO_SOLUTION)
    return next(iterator)


def test_max_plans_per_problem_stops_inc_plans_from_growing_further():
    iterator, first = iterator_with_plans(frontier_config(max_plans_per_problem=1))
    assert len(iterator.active_plans["p1"]) == 1

    second = refute(iterator)  # -> INC_COMPLEXITY (same state space)
    assert iterator.last_step == LastStep.INC_COMPLEXITY

    # p1 is already at the cap, so INC_PLANS has nothing to add for it and falls through to
    # complexity again instead.
    third = refute(iterator)
    assert iterator.last_step == LastStep.INC_COMPLEXITY
    assert len(iterator.active_plans["p1"]) == 1
    assert third["complexity"] == second["complexity"] + 1


def test_max_plans_per_problem_null_is_unbounded():
    # Default (no cap set) reproduces the pre-existing behaviour: INC_PLANS keeps adding.
    iterator, first = iterator_with_plans(frontier_config())
    refute(iterator)  # -> INC_COMPLEXITY
    refute(iterator)  # -> INC_PLANS
    assert iterator.last_step == LastStep.INC_PLANS
    assert len(iterator.active_plans["p1"]) == 2


class FakeCoverage:
    """Minimal stand-in for PlanStateCoverage: a plan is "new" iff its key hasn't been seen."""

    def __init__(self, always_new_after: int = 0):
        self.seen: set[tuple] = set()
        self.always_new_after = always_new_after
        self.calls = 0

    def add(self, problem_name, plan) -> bool:
        self.calls += 1
        key = (problem_name, plan_key(plan))
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


def test_inc_plans_skips_a_plan_the_coverage_tracker_rejects():
    # min_number_of_plans=1 already consumes "p1-0" for the initial batch (exempt from dedupe,
    # so it is kept regardless of what the coverage tracker says about it). The next plan
    # INC_PLANS would draw is "p1-1"; rig the fake tracker to reject exactly that one, as if it
    # added no state beyond what "p1-0" already covers.
    coverage = FakeCoverage()
    coverage.seen.add(("p1", plan_key(a_plan("p1-1"))))
    iterator, first = iterator_with_plans(frontier_config(), plan_coverage=coverage)
    assert iterator.active_plans["p1"] == [a_plan("p1-0")]
    refute(iterator)  # -> INC_COMPLEXITY
    # INC_PLANS is attempted next, but p1 is the only active problem and its offered plan is
    # rejected, so there is nothing to add; the ladder falls through to complexity again instead.
    third = refute(iterator)
    assert iterator.last_step == LastStep.INC_COMPLEXITY
    assert iterator.active_plans["p1"] == [a_plan("p1-0")]


def started_iterator(config, **kwargs):
    a, b = constants("a b")
    problems = [
        DummyProblem("p1", frozenset({Predicate("at", a)})),
        DummyProblem("p2", frozenset({Predicate("at", b)})),
    ]
    iterator = iter(ProblemIterator(problems, config, **kwargs))
    return iterator, next(iterator)


def test_frontier_expansion_is_deduped_by_state_coverage_not_just_action_sequence():
    coverage = FakeCoverage()
    iterator, _ = started_iterator(frontier_config(), plan_coverage=coverage)
    iterator.set_last_result(Result.FRONTIER)
    iterator.record_frontier_expansion({"p1": [a_plan("pick")]}, {})
    assert iterator.frontier_progress is True
    assert len(iterator.active_plans["p1"]) == 1

    # A different action sequence, but the fake coverage tracker treats it as covering nothing
    # new (simulating two distinct plans reaching the same states).
    coverage.seen.add(("p1", plan_key(a_plan("different-action"))))
    iterator.set_last_result(Result.FRONTIER)
    iterator.record_frontier_expansion({"p1": [a_plan("different-action")]}, {})
    assert iterator.frontier_progress is False
    assert len(iterator.active_plans["p1"]) == 1
