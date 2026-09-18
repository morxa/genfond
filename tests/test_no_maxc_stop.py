"""Tests for continuing the run past an exhausted complexity sweep (H33).

Reaching `max_complexity` on the current training set ends one sweep, not the run. Measured on
logistics_dp (47 problems, `--type datalog-sig`, 4 h graceful budget): the loop exhausted the
sweep after 28 rounds / 84 min with 15/47 solved and stopped with `failureReason=maxcomplexity`,
leaving 2.6 h unused, while slower variants that never reached the top of the sweep got 23/47 in
the same budget.

Everything here is iterator-level with stub round results, except the two end-to-end tests at the
bottom, which mock `solve_step` (as `tests/test_resample.py` does) to check the wiring between
`iterative_solver` and `ProblemIterator.__next__`: the resample budget, the shared resample
accounting with H32, and the `maxComplexityContinuations` stat.
"""

import logging

import pytest
from pddl.core import Plan
from pddl.logic import Constant, constants

from genfond import iterative_solver as isolver
from genfond.problem_iterator import (
    MAX_COST,
    MaxComplexityContinuation,
    ProblemIterator,
    Result,
    plan_key,
)

# --- helpers ----------------------------------------------------------------------------------


class DummyProblem:
    def __init__(self, name, objects=()):
        self.name = name
        self.objects = objects


def a_plan(*names):
    return Plan([(n, [Constant("x")]) for n in names])


def iterator_config(**overrides):
    config = {
        "min_complexity": 2,
        "max_complexity": 3,
        "use_selected_states": False,
        "use_unrestricted_features": False,
        "unselect_problems": False,
        "min_number_of_plans": 2,
        "max_frontier_expansions": 20,
        "reset_complexity_on_state_space_change": False,
        "resample_reset_complexity": False,
        "add_problem_after_success": False,
        "continue_after_max_complexity": True,
    }
    config.update(overrides)
    return config


def fresh_plans(problem):
    """A replacement stream for the resample, distinguishable from the original plans."""
    return iter([a_plan(f"{problem.name}-fresh-0"), a_plan(f"{problem.name}-fresh-1")])


def round_at(iterator, result=Result.NO_SOLUTION, cost=None):
    """Report `result` for the round that just ran and return the next round's configuration."""
    iterator.set_last_result(result, cost=cost)
    return next(iterator)


# --- (a) resample ------------------------------------------------------------------------------


def test_a_resample_continues_the_run_at_max_complexity():
    """(a) At the top of the sweep with a resample left, the run continues over a new sample of
    example plans and the sweep restarts at `min_complexity`: the resample changed the state
    space, so every refuted level is dropped, and the small instances at the bottom of the sweep
    are exactly what the exhausted big ones at the top could not be solved at."""
    config = iterator_config()
    problems = [DummyProblem("p1")]
    plans = {"p1": iter([a_plan("p1-0"), a_plan("p1-1")])}
    resamples = []

    def do_resample():
        resamples.append(True)
        return iterator.resample_planner_plans(fresh_plans)

    iterator = iter(
        ProblemIterator(
            problems,
            config,
            plans=plans,
            continuation=MaxComplexityContinuation(
                resample_available=lambda: len(resamples) < 1,
                resample=do_resample,
            ),
        )
    )

    assert next(iterator)["complexity"] == 2
    assert round_at(iterator)["complexity"] == 3  # the sweep climbs to max_complexity
    iterator.set_last_result(Result.NO_SOLUTION)
    assert iterator.refuted_complexity == 3

    continued = next(iterator)

    assert resamples == [True]
    assert iterator.max_complexity_continuations == 1
    # Restarted at min_complexity, with the level the sweep had reached remembered so the ladder
    # does not start adding example plans at the bottom of the restarted sweep.
    assert continued["complexity"] == config["min_complexity"]
    assert iterator.sweep_target == 3
    # The resample changed the state space, so nothing is refuted any more.
    assert iterator.refuted_complexity == config["min_complexity"] - 1
    # The sample really was replaced, and the training set was not touched.
    assert [plan_key(p) for p in iterator.active_plans["p1"]] == [
        plan_key(a_plan("p1-fresh-0")),
        plan_key(a_plan("p1-fresh-1")),
    ]
    assert [p.name for p in continued["active_problems"]] == ["p1"]


def test_a_resample_is_preferred_over_adding_a_problem():
    """Order matters: the resample changes the state space, which is the one escalation that can
    make a policy possible the exhausted sweep could not express at any complexity. Adding a
    problem is monotone, so it can only make the instance harder."""
    config = iterator_config()
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    plans = {p.name: iter([a_plan(f"{p.name}-0"), a_plan(f"{p.name}-1")]) for p in problems}
    resamples = []

    def do_resample():
        resamples.append(True)
        return iterator.resample_planner_plans(fresh_plans)

    iterator = iter(
        ProblemIterator(
            problems,
            config,
            plans=plans,
            continuation=MaxComplexityContinuation(resample_available=lambda: True, resample=do_resample),
        )
    )
    next(iterator)
    round_at(iterator)

    continued = round_at(iterator)

    assert resamples == [True]
    assert [p.name for p in continued["active_problems"]] == ["p1"]


def test_a_resample_that_replaces_nothing_falls_through_to_the_next_problem():
    """A resample whose streams can offer nothing new is not a continuation -- the next round
    would be identical to the one that just failed. Fall through to adding a problem instead."""
    config = iterator_config()
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    plans = {p.name: iter([a_plan(f"{p.name}-0"), a_plan(f"{p.name}-1")]) for p in problems}
    iterator = iter(
        ProblemIterator(
            problems,
            config,
            plans=plans,
            # No fresh stream and the problem's own stream is dry, so nothing is replaced.
            continuation=MaxComplexityContinuation(
                resample_available=lambda: True,
                resample=lambda: iterator.resample_planner_plans(),
            ),
        )
    )
    next(iterator)
    round_at(iterator)

    continued = round_at(iterator)

    assert [p.name for p in continued["active_problems"]] == ["p1", "p2"]
    assert iterator.max_complexity_continuations == 1


# --- (b) add the next problem --------------------------------------------------------------------


def a_three_problem_iterator():
    """Three problems, no plan streams (so the INC_PLANS branch is out of the picture), and no
    resample available -- the setup in which the continuation can only add a problem."""
    config = iterator_config(max_complexity=4)
    problems = [DummyProblem(name) for name in ("p1", "p2", "p3")]
    return iter(ProblemIterator(problems, config))


def test_b_the_next_problem_is_added_and_the_sweep_restarts():
    """(b) With the resample budget spent (here: no resample at all) and unsolved problems left
    outside the training set, the next one joins it through the normal add-a-problem bookkeeping
    and the sweep restarts at `min_complexity` instead of at `succ_complexity`.

    Restarting is sound, not merely cheap: the feature pool at complexity c contains every
    feature of complexity < c, so no policy reachable from the top of the sweep becomes
    unreachable from the bottom. It can only cost time -- and it buys far smaller instances,
    which is the whole point when the large ones at the top produced nothing.
    """
    iterator = a_three_problem_iterator()

    assert next(iterator)["complexity"] == 2
    # A success at complexity 3 sets succ_complexity, which is where the add-a-problem branch
    # would normally resume the sweep.
    assert round_at(iterator)["complexity"] == 3
    assert round_at(iterator, Result.SUCCESS, cost=(10,))["complexity"] == 4
    assert iterator.succ_complexity == 3
    # The ordinary add-a-problem branch: the training set is solved, so p2 joins it at
    # succ_complexity and this is *not* a continuation.
    resumed = round_at(iterator)
    assert resumed["complexity"] == 3
    assert [p.name for p in resumed["active_problems"]] == ["p1", "p2"]
    assert iterator.max_complexity_continuations == 0
    # p1+p2 are never solved, so the sweep now runs out at max_complexity with p3 still outside.
    assert round_at(iterator)["complexity"] == 4

    continued = round_at(iterator)

    assert iterator.max_complexity_continuations == 1
    assert [p.name for p in continued["active_problems"]] == ["p1", "p2", "p3"]
    assert continued["complexity"] == 2  # min_complexity, not succ_complexity (3)
    # Adding an instance is monotone, so the bound the last success established survives -- the
    # one refutation rule the add-a-problem branch is entitled to keep.
    assert iterator.refuted_complexity == iterator.succ_complexity - 1 == 2


# --- (c) nothing left, (d) switched off ----------------------------------------------------------


def test_c_stop_iteration_when_nothing_is_left_to_continue_with():
    """(c) Every unsolved problem already in the training set and no resample left: the run ends
    exactly as it did before this mechanism existed."""
    iterator = iter(ProblemIterator([DummyProblem("p1")], iterator_config()))

    assert next(iterator)["complexity"] == 2
    assert round_at(iterator)["complexity"] == 3
    iterator.set_last_result(Result.NO_SOLUTION)
    with pytest.raises(StopIteration):
        next(iterator)
    assert iterator.max_complexity_continuations == 0


def test_c_the_wall_deadline_stops_the_run_even_with_continuations_available():
    """A continuation costs planner calls and grows the state space, so it must not be spent on a
    run the round loop is about to stop anyway."""
    config = iterator_config()
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    iterator = iter(
        ProblemIterator(
            problems,
            config,
            continuation=MaxComplexityContinuation(
                time_left=lambda: False,
                resample_available=lambda: True,
                resample=lambda: (2, 1),
            ),
        )
    )

    next(iterator)
    round_at(iterator)
    iterator.set_last_result(Result.NO_SOLUTION)
    with pytest.raises(StopIteration):
        next(iterator)
    assert iterator.max_complexity_continuations == 0


def test_d_the_option_off_reproduces_the_old_behaviour():
    """(d) Off, the iterator stops at the top of the sweep even though both continuations are
    available."""
    config = iterator_config(continue_after_max_complexity=False)
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    iterator = iter(
        ProblemIterator(
            problems,
            config,
            continuation=MaxComplexityContinuation(resample_available=lambda: True, resample=lambda: (2, 1)),
        )
    )

    next(iterator)
    round_at(iterator)
    iterator.set_last_result(Result.NO_SOLUTION)
    with pytest.raises(StopIteration):
        next(iterator)
    assert iterator.max_complexity_continuations == 0


def test_the_continuation_logs_one_line_per_continuation(caplog):
    """One INFO line per continuation, naming what the run is continuing by."""
    config = iterator_config()
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    iterator = iter(ProblemIterator(problems, config))
    next(iterator)
    round_at(iterator)

    with caplog.at_level(logging.INFO, logger="genfond.problem_iterator"):
        round_at(iterator)

    lines = [record.getMessage() for record in caplog.records]
    assert "Max complexity 3 reached with 2 unsolved problem(s); continuing by adding problem p2" in lines


# --- (e) max_cost --------------------------------------------------------------------------------


def test_e_max_cost_is_not_loosened_by_a_continuation():
    """(e) A continuation is not permission to accept a worse policy: the best policy so far
    still stands and the next round is still asked to beat its cost.

    The resample leaves `max_cost` alone by design. The add-a-problem branch does reset it to
    `MAX_COST`, but that is a no-op on this path: reaching the continuation with a problem left
    to add means the last elif of `__next__` failed on `active_problems_solved`, and a training
    set unsolved since the last problem joined it already carries `MAX_COST`.
    """
    config = iterator_config()
    problems = [DummyProblem("p1")]
    plans = {"p1": iter([a_plan("p1-0"), a_plan("p1-1")])}
    iterator = iter(
        ProblemIterator(
            problems,
            config,
            plans=plans,
            continuation=MaxComplexityContinuation(
                resample_available=lambda: True,
                resample=lambda: iterator.resample_planner_plans(fresh_plans),
            ),
        )
    )

    next(iterator)
    # A success tightens max_cost; p1 is never marked solved, so the run keeps going.
    climbed = round_at(iterator, Result.SUCCESS, cost=(10,))
    assert climbed["max_cost"] == 9

    continued = round_at(iterator)

    assert iterator.max_complexity_continuations == 1
    assert continued["complexity"] == 2
    assert continued["max_cost"] == 9
    assert iterator.max_cost == 9


def test_e_max_cost_stays_at_max_cost_across_an_add_problem_continuation():
    """The other half of (e): the add-a-problem continuation neither tightens nor loosens."""
    iterator = a_three_problem_iterator()
    next(iterator)
    round_at(iterator)
    round_at(iterator, Result.SUCCESS, cost=(10,))
    round_at(iterator)  # p2 joins through the ordinary branch, which resets max_cost
    assert iterator.max_cost == MAX_COST
    round_at(iterator)

    continued = round_at(iterator)

    assert iterator.max_complexity_continuations == 1
    assert continued["max_cost"] == MAX_COST


# --- end to end through solve_iteratively --------------------------------------------------------

A, B, C = constants("a b c")
CYCLE = [("pick", [A, B]), ("put", [A, C]), ("pick", [A, C]), ("put", [A, B])]


def pool_plan(n):
    """The first `n` actions of the pick/put cycle -- a genuinely executable plan for the
    `simple_blocks` fixture, since `StateSpaceGraph` replays every example plan for real."""
    return Plan([CYCLE[i % len(CYCLE)] for i in range(1, n + 1)])


def solver_config(**overrides):
    config = {
        "min_complexity": 2,
        "max_complexity": 3,
        "use_unrestricted_features": False,
        "reset_complexity_on_state_space_change": False,
        "resample_reset_complexity": False,
        "add_problem_after_success": False,
        "unselect_problems": False,
        "min_number_of_plans": 2,
        "max_plans_per_problem": None,
        "max_frontier_expansions": 20,
        "use_example_plans": True,
        "frontier_expansion": False,
        "policy_conformant_plans": False,
        "policy_prefix_plans": False,
        "policy_iterations": 1,
        "validation_iterations": 1,
        "validation_max_consecutive_failures": None,
        "validation_time_limit": None,
        "stop_after_first_solution": True,
        "final_cost_minimization": False,
        "minimize_good_signatures": "none",
        "minimize_selected_count": "none",
        "keep_best_policy": True,
        "planner": "siw",
        "planners": {"siw": {"seed": 0, "restarts": 1}},
        "seed": 7,
        "resample_on_stall": True,
        # High enough that H32's own stall trigger never fires: what these tests exercise is the
        # max-complexity continuation reaching for the same resample.
        "stall_rounds": 99,
        "resample_max": 1,
        "continue_after_max_complexity": True,
    }
    config.update(overrides)
    return config


def run_failing_rounds(monkeypatch, simple_blocks, **config_overrides):
    """A run whose every round fails, so it climbs straight to max_complexity and stops there.

    Each planner call hands out a fresh two-plan stream from a disjoint slice of the plan pool
    (two plans is exactly the `min_number_of_plans` floor, so a stream is always dry by the time
    anything wants more), which makes it visible in the plans themselves which stream they came
    from. Returns `(stats, planner configs, number of rounds)`.
    """
    domain, problem = simple_blocks
    planner_configs: list[dict] = []

    def fake_compute_plans(domain_str, problem_str, planner_config):
        index = len(planner_configs)
        planner_configs.append(dict(planner_config))
        return iter([pool_plan(2 * index + 1), pool_plan(2 * index + 2)])

    rounds: list[int] = []

    def fake_solve_step(**kwargs):
        rounds.append(kwargs["complexity"])
        return (Result.NO_SOLUTION, None, [])

    monkeypatch.setattr(
        isolver,
        "_get_example_plan_computer",
        lambda config: (fake_compute_plans, dict(config["planners"]["siw"]), "siw"),
    )
    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)

    _, _, stats = isolver.solve_iteratively(domain, [problem], solver_config(**config_overrides))
    return stats, planner_configs, rounds


def test_end_to_end_the_run_continues_past_max_complexity_by_resampling(monkeypatch, simple_blocks):
    """The wiring: `iterative_solver` hands the iterator a continuation that spends H32's
    resample budget and accounts for it in the same stats, and the run keeps going past the top
    of the sweep instead of stopping there."""
    stats, planner_configs, rounds = run_failing_rounds(monkeypatch, simple_blocks)

    # Without the continuation the run is complexity 2, 3 and done; with it the sweep restarts.
    assert rounds == [2, 3, 2, 3]
    assert stats["maxComplexityContinuations"] == 1
    # The same resample as H32's, accounted for in the same counters.
    assert stats["resamples"] == 1
    assert stats["resampledPlans"] == 2
    assert planner_configs == [{"seed": 0, "restarts": 1}, {"seed": 1000003, "restarts": 2}]
    # The run still ends by exhausting the iterator once nothing is left to continue with, and
    # still without a policy. (`failureReason` is not asserted here: it is only ever written by
    # the round that *produced* a policy which failed to solve everything, so a run whose every
    # round is NO_SOLUTION never sets it -- with or without this mechanism.)
    assert "stoppedBy" not in stats


def test_end_to_end_the_option_off_stops_at_max_complexity(monkeypatch, simple_blocks):
    """(d) end to end: off, the run stops at the top of the sweep with the resample unspent."""
    stats, planner_configs, rounds = run_failing_rounds(
        monkeypatch, simple_blocks, continue_after_max_complexity=False
    )

    assert rounds == [2, 3]
    assert "maxComplexityContinuations" not in stats
    assert "resamples" not in stats
    assert planner_configs == [{"seed": 0, "restarts": 1}]
    assert "stoppedBy" not in stats
