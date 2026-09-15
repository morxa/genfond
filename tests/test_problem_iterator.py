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
    objects: tuple = ()


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
        "max_frontier_expansions": 20,
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
        "reset_complexity_on_state_space_change": False,
        "add_problem_after_success": False,
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
        "max_frontier_expansions": 20,
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


def iterator_with_plans(config, num_plans=10):
    a, b = constants("a b")
    problems = [
        DummyProblem("p1", frozenset({Predicate("at", a)})),
        DummyProblem("p2", frozenset({Predicate("at", b)})),
    ]
    plans = {p.name: iter([a_plan(f"{p.name}-{i}") for i in range(num_plans)]) for p in problems}
    iterator = iter(ProblemIterator(problems, config, plans=plans))
    return iterator, next(iterator)


def refute(iterator):
    """Report the current round as unsatisfiable and take the next one."""
    iterator.set_last_result(Result.NO_SOLUTION)
    return next(iterator)


def test_enforcement_is_dropped_after_a_plan_is_added():
    # min_feature_complexity(c) is only justified by "c-1 was refuted", which is a statement
    # about one particular state space. Adding a plan changes it, and the enlarged state space
    # may well admit a policy below c that the constraint would exclude.
    iterator, first = iterator_with_plans(frontier_config())
    assert first["enforce_highest_complexity"] is True

    second = refute(iterator)  # -> INC_COMPLEXITY, same state space
    assert iterator.last_step == LastStep.INC_COMPLEXITY
    assert second["enforce_highest_complexity"] is True

    third = refute(iterator)  # -> INC_PLANS, state space changed
    assert iterator.last_step == LastStep.INC_PLANS
    assert third["complexity"] == second["complexity"]
    assert third["enforce_highest_complexity"] is False

    # Refuting the level again over the new state space re-establishes the bound.
    fourth = refute(iterator)
    assert fourth["complexity"] == third["complexity"] + 1
    assert fourth["enforce_highest_complexity"] is True


def test_enforcement_is_dropped_after_a_frontier_expansion():
    iterator, _ = iterator_with_plans(frontier_config())
    # At min_complexity the bound holds by definition, so climb one level first.
    second = refute(iterator)
    assert second["complexity"] > frontier_config()["min_complexity"]
    assert second["enforce_highest_complexity"] is True

    iterator.set_last_result(Result.FRONTIER)
    iterator.record_frontier_expansion({"p1": [a_plan("pick")]}, {})
    third = next(iterator)
    assert iterator.last_step == LastStep.EXPAND_FRONTIER
    assert third["enforce_highest_complexity"] is False


def test_a_restricted_round_refutes_nothing():
    # The restricted generators are a subset of the unrestricted ones, so their failure says
    # nothing about a pool that has not been tried yet.
    iterator, first = iterator_with_plans(frontier_config(use_unrestricted_features=True))
    assert first["all_features"] is False
    iterator.set_last_result(Result.NO_SOLUTION)
    assert iterator.refuted_complexity < first["complexity"]

    unrestricted = next(iterator)
    assert unrestricted["all_features"] is True
    assert unrestricted["complexity"] == first["complexity"]
    iterator.set_last_result(Result.NO_SOLUTION)
    assert iterator.refuted_complexity == first["complexity"]


def test_enforcement_survives_an_unchanged_state_space():
    # Toggling the generators does not touch the state space, so the refutations stay valid
    # and the optimization must not be given up.
    iterator, first = iterator_with_plans(frontier_config(use_unrestricted_features=True))
    refute(iterator)  # -> unrestricted, same complexity
    higher = refute(iterator)
    assert higher["complexity"] == first["complexity"] + 1
    assert higher["enforce_highest_complexity"] is True


def test_success_refutes_its_own_complexity():
    # clingo minimizes the feature cost, so a success means nothing at this complexity beats
    # the new max_cost. Keeping the bound is what lets solve() skip the rounds it now knows
    # are unsatisfiable instead of grounding them.
    iterator, first = iterator_with_plans(frontier_config())
    iterator.set_last_result(Result.SUCCESS, cost=(9,))
    iterator.set_solved(iterator.problems[0])
    second = next(iterator)
    assert second["complexity"] == first["complexity"] + 1
    assert second["enforce_highest_complexity"] is True


def test_complexity_is_not_restarted_by_default():
    iterator, first = iterator_with_plans(frontier_config())
    refute(iterator)
    third = refute(iterator)
    assert iterator.last_step == LastStep.INC_PLANS
    assert third["complexity"] == first["complexity"] + 1


def test_complexity_restarts_at_min_complexity_when_configured():
    config = frontier_config(reset_complexity_on_state_space_change=True)
    iterator, first = iterator_with_plans(config)
    refute(iterator)
    third = refute(iterator)
    assert iterator.last_step == LastStep.INC_PLANS
    assert third["complexity"] == config["min_complexity"]
    # Restarting re-establishes the refutations from the bottom, so the bound is valid again.
    assert third["enforce_highest_complexity"] is True


def test_restarted_sweep_gets_one_level_deeper_per_plan():
    # Without the sweep_target guard, "add a plan" and "restart at min_complexity" alternate
    # at one fixed level and the search never reaches the higher complexities at all. Each
    # plan must instead be added one level deeper than the last, as it is without the restart.
    config = frontier_config(reset_complexity_on_state_space_change=True, max_complexity=6)
    iterator, first = iterator_with_plans(config)
    sweep = [(first["complexity"], len(iterator.active_plans["p1"]))]
    for _ in range(14):
        iteration = refute(iterator)
        sweep.append((iteration["complexity"], len(iterator.active_plans["p1"])))
    # Sweep 2..3 with one plan, restart and sweep 2..4 with two, and so on: the plan is always
    # added at the deepest level reached so far and the sweep then restarts at min_complexity.
    assert sweep == [
        (2, 1),
        (3, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (2, 4),
        (3, 4),
        (4, 4),
        (5, 4),
        (6, 4),
        (2, 5),
    ]


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


def test_add_problem_after_success_skips_the_complexity_climb():
    # iterative_solver.py calls set_last_result(SUCCESS, cost=...) when the ASP round
    # succeeds, then -- if execute_policy fails on a problem outside the training set --
    # set_last_result(NO_SOLUTION) again, without an intervening next(), to report the round
    # as not yet a full policy. With the switch on, that must add the next unsolved problem
    # right away instead of climbing complexity on the same training set.
    iterator, first = started_iterator(frontier_config(add_problem_after_success=True))
    assert first["active_problems"] == [iterator.problems[0]]

    iterator.set_last_result(Result.SUCCESS, cost=(9,))
    iterator.set_solved(iterator.problems[0])
    iterator.set_last_result(Result.NO_SOLUTION)
    second = next(iterator)

    assert second["active_problems"] == iterator.problems
    assert second["max_cost"] == MAX_COST


def test_without_the_switch_success_climbs_complexity_on_the_same_problems():
    # Same scenario, switch off: today's behaviour is unchanged -- the round after a success
    # climbs feature complexity on the same training set instead of adding the next problem.
    iterator, first = started_iterator(frontier_config())
    assert first["active_problems"] == [iterator.problems[0]]

    iterator.set_last_result(Result.SUCCESS, cost=(9,))
    iterator.set_solved(iterator.problems[0])
    iterator.set_last_result(Result.NO_SOLUTION)
    second = next(iterator)

    assert second["active_problems"] == first["active_problems"]
    assert second["complexity"] == first["complexity"] + 1
    assert iterator.last_step == LastStep.INC_COMPLEXITY


def test_a_success_that_was_not_proven_optimal_refutes_nothing():
    # Under a solve_time_limit a success can be a model that merely satisfies every constraint.
    # Its cost is an upper bound, so it is still worth beating (max_cost tightens), but nothing
    # rules out a cheaper policy at this very complexity -- so the level must not be refuted and
    # min_feature_complexity must stay off.
    iterator, first = iterator_with_plans(frontier_config())
    iterator.set_last_result(Result.SUCCESS, cost=(9,), optimal=False)
    assert iterator.max_cost == 8
    assert iterator.refuted_complexity < first["complexity"]
    iterator.set_solved(iterator.problems[0])
    second = next(iterator)
    assert second["complexity"] == first["complexity"] + 1
    assert second["enforce_highest_complexity"] is False


def test_min_train_objects_starts_the_training_set_above_the_threshold():
    # p1 is smaller than the threshold, so the training set must skip straight to p2 even
    # though p1 is the first candidate in object-count order.
    a, b = constants("a b")
    problems = [
        DummyProblem("p1", frozenset({Predicate("at", a)}), objects=(a,)),
        DummyProblem("p2", frozenset({Predicate("at", b)}), objects=(a, b)),
    ]
    config = frontier_config(min_train_objects=2)
    iterator = iter(ProblemIterator(problems, config))
    first = next(iterator)
    assert first["active_problems"] == [problems[1]]
    # p1 stays a live obligation: a success is still checked against it, and the run keeps
    # going until it (or a policy that happens to solve it) is accounted for.
    assert iterator.solved["p1"] is False
    assert iterator._next_addable_problem() is None  # nothing else clears the threshold


def test_min_train_objects_never_adds_a_below_threshold_problem_later_either():
    # add_problem_after_success drives the same _next_addable_problem/_add_next_problem path;
    # a below-threshold problem must stay excluded there too.
    a, b, c = constants("a b c")
    problems = [
        DummyProblem("p1", frozenset({Predicate("at", a)}), objects=(a,)),
        DummyProblem("p2", frozenset({Predicate("at", b)}), objects=(a, b)),
        DummyProblem("p3", frozenset({Predicate("at", c)}), objects=(a, b, c)),
    ]
    config = frontier_config(min_train_objects=2, add_problem_after_success=True)
    iterator = iter(ProblemIterator(problems, config))
    first = next(iterator)
    assert first["active_problems"] == [problems[1]]

    iterator.set_last_result(Result.SUCCESS, cost=(9,))
    iterator.set_solved(problems[1])
    second = next(iterator)
    assert second["active_problems"] == [problems[1], problems[2]]
    assert problems[0] not in second["active_problems"]


def test_min_train_objects_falls_back_when_no_problem_meets_it():
    # A threshold above every problem's object count would otherwise leave
    # _next_addable_problem returning None forever, so __next__ would raise StopIteration
    # before a single problem is ever added. Falling back to the old behaviour instead.
    a, b = constants("a b")
    problems = [
        DummyProblem("p1", frozenset({Predicate("at", a)}), objects=(a,)),
        DummyProblem("p2", frozenset({Predicate("at", b)}), objects=(a, b)),
    ]
    config = frontier_config(min_train_objects=10)
    iterator = iter(ProblemIterator(problems, config))
    assert iterator.min_train_objects is None
    first = next(iterator)
    assert first["active_problems"] == [problems[0]]


def test_min_train_objects_null_is_byte_identical_to_unset():
    a, b = constants("a b")
    problems = [
        DummyProblem("p1", frozenset({Predicate("at", a)}), objects=(a,)),
        DummyProblem("p2", frozenset({Predicate("at", b)}), objects=(a, b)),
    ]
    config = frontier_config(min_train_objects=None)
    iterator = iter(ProblemIterator(problems, config))
    assert iterator.min_train_objects is None
    first = next(iterator)
    assert first["active_problems"] == [problems[0]]


def test_a_timeout_refutes_nothing_but_still_escalates():
    # A solve that ran out of its budget without a model says nothing about the complexity
    # level; treating it like NO_SOLUTION would wrongly enable min_feature_complexity later.
    iterator, first = iterator_with_plans(frontier_config(use_unrestricted_features=False))
    iterator.set_last_result(Result.TIMEOUT)
    assert iterator.refuted_complexity < first["complexity"]
    second = next(iterator)
    assert second["enforce_highest_complexity"] is False

    # Escalation is otherwise identical to NO_SOLUTION; only the refutation differs.
    refuting, _ = iterator_with_plans(frontier_config(use_unrestricted_features=False))
    refuting.set_last_result(Result.NO_SOLUTION)
    after_no_solution = next(refuting)
    assert iterator.last_step == refuting.last_step
    assert second["complexity"] == after_no_solution["complexity"]
    assert after_no_solution["enforce_highest_complexity"] is True
