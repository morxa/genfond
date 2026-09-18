"""Tests for deferring problems the planner cannot plan for, and for bounding the expansion (H25).

`StateSpaceGraph` is only plan-restricted while there are plans to restrict it to: a problem with
no example plan falls through to the *unrestricted* expansion of the whole instance. The problems
that have no example plan are exactly the ones the planner could not solve, i.e. the hard ones --
measured on sokoban with `--type datalog-sig`, SIW finds no plan for the smallest problem, the
first round starts with that problem alone, and the feature-pool stage never returns (12 h and
4 h runs, repeatedly). Neither the solve budget nor the graceful stop reaches that phase, so the
run produces nothing at all.

The iterator-level tests run the real `ProblemIterator` against stub problems and plan streams;
the end-to-end ones mock `solve_step` and the planner (as `tests/test_resample.py` does) but run
the real iterator, so the deferral bookkeeping is genuinely exercised.
"""

import pytest
from pddl.core import Plan
from pddl.logic import Constant, constants

from genfond import iterative_solver as isolver
from genfond.problem_iterator import ProblemIterator, Result
from genfond.shutdown import request_stop, reset_stop
from genfond.state_space_generator import (
    ExpansionInterrupted,
    ExpansionLimitExceeded,
    StateSpaceGraph,
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
        "max_complexity": 7,
        "use_selected_states": False,
        "use_unrestricted_features": False,
        "unselect_problems": False,
        "min_number_of_plans": 2,
        "max_frontier_expansions": 20,
        "reset_complexity_on_state_space_change": False,
        "resample_reset_complexity": False,
        # On, so the add-a-problem branch (the one that resolves a candidate, and with it the
        # deferral) is reached right after a success instead of only at the top of the sweep.
        "add_problem_after_success": True,
        "continue_after_max_complexity": False,
        "defer_planless_problems": True,
        "planless_retries": 2,
    }
    config.update(overrides)
    return config


def an_iterator(config, streams, fresh=None):
    """An iterator over one problem per entry of `streams` (a name -> list-of-plans mapping)."""
    problems = [DummyProblem(name) for name in streams]
    plans = {name: iter(list(stream)) for name, stream in streams.items()}
    return iter(ProblemIterator(problems, config, plans=plans, fresh_plan_iterator=fresh))


# --- (a) a planless problem is deferred and the next one is added -------------------------------


def test_a_planless_problem_is_deferred_and_the_next_one_is_added():
    """The planner has nothing for p1, so p1 must not enter the training set -- the round would
    be built over its unrestricted state space. p2 is added in its place."""
    iterator = an_iterator(iterator_config(), {"p1": [], "p2": [a_plan("p2-0"), a_plan("p2-1")]})

    kwargs = next(iterator)

    assert [p.name for p in kwargs["active_problems"]] == ["p2"]
    assert iterator.deferred_problems == {"p1"}
    assert len(iterator.active_plans["p2"]) == 2


def test_a_deferred_problem_stays_in_the_evaluation_set():
    """Deferring is about *training* only: the problem still counts as unsolved, so every
    candidate policy is still executed on it and the run is still not finished without it."""
    iterator = an_iterator(iterator_config(), {"p1": [], "p2": [a_plan("p2-0")]})
    next(iterator)

    assert iterator.solved["p1"] is False
    assert "p1" in [p.name for p in iterator.get_unsolved_problems()]


def test_a_problem_that_only_has_policy_plans_is_not_planless():
    """A trajectory recorded from a working policy restricts the state space just as well as a
    planner plan, so a problem holding one is addable even though its stream is empty."""
    iterator = an_iterator(iterator_config(), {"p1": [], "p2": [a_plan("p2-0")]})
    iterator.record_policy_plans({"p1": a_plan("policy-step")})

    kwargs = next(iterator)

    assert [p.name for p in kwargs["active_problems"]] == ["p1"]
    assert iterator.deferred_problems == set()


# --- (b) retries against a reseeded stream ------------------------------------------------------


def test_a_deferred_problem_is_retried_with_a_fresh_stream_up_to_planless_retries():
    """`planless_retries: 2` means the initial attempt plus two reseeded retries. After that the
    problem is skipped permanently for training -- each retry is a full planner run on a problem
    the planner has already failed on."""
    fresh_calls = []

    def fresh(problem, attempt):
        fresh_calls.append((problem.name, attempt))
        return iter([])

    iterator = an_iterator(
        iterator_config(),
        {"p1": [], "p2": [a_plan("p2-0")], "p3": [a_plan("p3-0")]},
        fresh=fresh,
    )
    # Round 1 defers p1 (attempt 1, against its own stream) and adds p2.
    next(iterator)
    assert fresh_calls == []
    # Every later round that looks for a problem to add retries p1 first, on a fresh stream.
    iterator.set_last_result(Result.SUCCESS, cost=(4,))
    next(iterator)
    assert [name for name, _ in fresh_calls] == ["p1"]
    iterator.set_last_result(Result.SUCCESS, cost=(4,))
    next(iterator)

    assert fresh_calls == [("p1", 1), ("p1", 2)]
    assert iterator.planless_retries_spent == 2
    assert iterator.planless_attempts["p1"] == 3
    assert "p1" in iterator.skipped_planless


def test_a_permanently_skipped_problem_is_never_offered_again():
    """Once skipped, p1 must not be resolved as addable -- `_add_next_problem` asserts there is a
    problem to add, so a candidate that can never be added has to disappear from the scan."""
    iterator = an_iterator(iterator_config(planless_retries=0), {"p1": [], "p2": [a_plan("p2-0")]})
    next(iterator)

    assert "p1" in iterator.skipped_planless
    assert iterator._next_addable_problem() is None
    # ... and the run ends cleanly rather than asserting.
    iterator.set_last_result(Result.SUCCESS, cost=(4,))
    with pytest.raises(StopIteration):
        for _ in range(10):
            next(iterator)
            iterator.set_last_result(Result.SUCCESS, cost=(4,))


def test_a_retry_that_finds_a_plan_adds_the_problem():
    """The retry exists because SIW is randomized: a reseeded stream with permuted views can find
    a plan where the first one did not."""
    iterator = an_iterator(
        iterator_config(),
        {"p1": [], "p2": [a_plan("p2-0")]},
        fresh=lambda problem, attempt: iter([a_plan("fresh-p1")]),
    )
    next(iterator)
    assert iterator.deferred_problems == {"p1"}

    iterator.set_last_result(Result.SUCCESS, cost=(4,))
    kwargs = next(iterator)

    assert [p.name for p in kwargs["active_problems"]] == ["p2", "p1"]
    assert [str(p.actions[0][0]) for p in iterator.active_plans["p1"]] == ["fresh-p1"]
    assert "p1" not in iterator.skipped_planless


# --- (c) everything planless -> a clean stop ----------------------------------------------------


def test_every_problem_planless_stops_the_iteration_immediately():
    iterator = an_iterator(iterator_config(), {"p1": [], "p2": []})

    with pytest.raises(StopIteration):
        next(iterator)

    assert iterator.deferred_problems == {"p1", "p2"}
    assert iterator.stopped_without_training_problems() is True


# --- (d) / the expansion bound ------------------------------------------------------------------


def test_max_states_stops_the_expansion(simple_blocks):
    """The bound is checked per expanded state, so the count may overshoot by one state's
    successors -- it is a safety net against an expansion that never finishes, not a budget."""
    domain, problem = simple_blocks
    full = StateSpaceGraph(domain, problem, prune=False)
    assert len(full.nodes) > 3

    with pytest.raises(ExpansionLimitExceeded) as excinfo:
        StateSpaceGraph(domain, problem, prune=False, max_states=3)

    assert excinfo.value.problem_name == problem.name
    assert excinfo.value.max_states == 3


def test_max_states_none_expands_everything(simple_blocks):
    domain, problem = simple_blocks
    bounded = StateSpaceGraph(domain, problem, prune=False, max_states=None)
    assert len(bounded.nodes) == len(StateSpaceGraph(domain, problem, prune=False).nodes)


def test_defer_active_problem_shrinks_the_training_set_and_retries():
    """What `iterative_solver` does on an `ExpansionLimitExceeded`: drop the offending problem and
    run the identical configuration over what is left, with nothing escalated."""
    iterator = an_iterator(iterator_config(), {"p1": [a_plan("p1-0")], "p2": [a_plan("p2-0")], "p3": [a_plan("p3-0")]})
    next(iterator)
    iterator.set_last_result(Result.SUCCESS, cost=(4,))
    kwargs = next(iterator)
    assert [p.name for p in kwargs["active_problems"]] == ["p1", "p2"]
    complexity_before = iterator.complexity
    # As if the rounds over the two-problem set had refuted every level up to here.
    iterator.refuted_complexity = 5

    iterator.defer_active_problem(iterator.problems[0], expansion_cutoff=True)
    iterator.set_last_result(Result.DEFERRED)
    kwargs = next(iterator)

    assert [p.name for p in kwargs["active_problems"]] == ["p2"]
    assert kwargs["complexity"] == complexity_before
    assert iterator.expansion_cutoffs == 1
    assert iterator.deferred_problems == {"p1"}
    # Not retried later: every way the run can change a problem's state space only ever adds
    # plans, so an expansion cut off at N states would only be cut off again.
    assert "p1" in iterator.skipped_planless
    # The training set shrank, which is not a monotone change: a level refuted over the larger
    # set says nothing about the smaller, easier one, so the bound goes back to the floor.
    assert iterator.refuted_complexity == iterator.config["min_complexity"] - 1


def test_deferring_the_last_training_problem_adds_the_next_one():
    iterator = an_iterator(iterator_config(), {"p1": [a_plan("p1-0")], "p2": [a_plan("p2-0")]})
    kwargs = next(iterator)
    assert [p.name for p in kwargs["active_problems"]] == ["p1"]

    iterator.defer_active_problem(iterator.problems[0], expansion_cutoff=True)
    iterator.set_last_result(Result.DEFERRED)
    kwargs = next(iterator)

    assert [p.name for p in kwargs["active_problems"]] == ["p2"]


# --- (e) the stop flag interrupts the expansion loop --------------------------------------------


def test_the_stop_flag_interrupts_the_expansion(monkeypatch, simple_blocks):
    """A multi-hour unrestricted expansion used to swallow SIGTERM entirely: the flag was first
    polled in `Solver.solve`, which such a round never reaches. The poll interval is shortened
    here so the fixture does not have to be big enough to reach the real one."""
    domain, problem = simple_blocks
    monkeypatch.setattr("genfond.state_space_generator.STOP_CHECK_INTERVAL", 1)
    reset_stop()
    request_stop()
    try:
        with pytest.raises(ExpansionInterrupted):
            StateSpaceGraph(domain, problem, prune=False)
    finally:
        reset_stop()


def test_the_expansion_is_not_interrupted_without_a_stop_request(monkeypatch, simple_blocks):
    domain, problem = simple_blocks
    monkeypatch.setattr("genfond.state_space_generator.STOP_CHECK_INTERVAL", 1)
    reset_stop()
    assert StateSpaceGraph(domain, problem, prune=False).nodes


# --- (f) the option off reproduces the old behaviour --------------------------------------------


def test_with_the_option_off_a_planless_problem_is_added_as_before():
    """Which is what makes the round expand it unrestrictedly -- `plans=[]` switches off every
    plan-restriction branch in `StateSpaceGraph`."""
    iterator = an_iterator(iterator_config(defer_planless_problems=False), {"p1": [], "p2": [a_plan("p2-0")]})

    kwargs = next(iterator)

    assert [p.name for p in kwargs["active_problems"]] == ["p1"]
    assert iterator.active_plans["p1"] == []
    assert iterator.deferred_problems == set()


# --- end to end through solve_iteratively -------------------------------------------------------

A, B, C = constants("a b c")
CYCLE = [("pick", [A, B]), ("put", [A, C]), ("pick", [A, C]), ("put", [A, B])]


def pool_plan(n):
    """The first `n` actions of the pick/put cycle, applicable in the `simple_blocks` fixture."""
    return Plan([CYCLE[i % len(CYCLE)] for i in range(1, n + 1)])


def a_second_problem(domain, problem):
    """A copy of the fixture problem under a different name, so a suite has two of them."""
    return type(problem)(
        "p2",
        domain=domain,
        requirements=problem.requirements,
        objects=list(problem.objects),
        init=list(problem.init),
        goal=problem.goal,
    )


def solver_config(**overrides):
    config = {
        "min_complexity": 2,
        "max_complexity": 40,
        "use_unrestricted_features": False,
        "reset_complexity_on_state_space_change": False,
        "resample_reset_complexity": False,
        "add_problem_after_success": False,
        "unselect_problems": False,
        "min_number_of_plans": 1,
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
        "resample_on_stall": False,
        "stall_rounds": 6,
        "resample_max": 0,
        "continue_after_max_complexity": False,
        "defer_planless_problems": True,
        "planless_retries": 2,
    }
    config.update(overrides)
    return config


def test_no_problem_with_a_plan_stops_the_run_with_a_reason(monkeypatch, simple_blocks):
    """(c) end to end: the planner finds nothing for the only problem, so no round is ever built
    -- the unrestricted expansion this replaces never returned at all."""
    domain, problem = simple_blocks
    solve_calls = []

    monkeypatch.setattr(
        isolver,
        "_get_example_plan_computer",
        lambda config: ((lambda d, p, c: iter([])), dict(config["planners"]["siw"]), "siw"),
    )
    monkeypatch.setattr(
        isolver,
        "solve_step",
        lambda **kwargs: solve_calls.append(kwargs) or (Result.NO_SOLUTION, None, []),
    )

    policy, solved, stats = isolver.solve_iteratively(domain, [problem], solver_config())

    assert policy is None
    assert solved == []
    assert solve_calls == []
    assert stats["failureReason"] == "no_example_plans"
    assert stats["deferredProblems"] == 1
    # With nothing else to train on the run ends after the first attempt, so no retry is spent;
    # `test_the_retry_stream_is_reseeded_like_a_resample` covers the retry itself.
    assert "planlessRetries" not in stats


def test_the_retry_stream_is_reseeded_like_a_resample(monkeypatch, simple_blocks):
    """(b) end to end: SIW's restart 1 is the identity view of the task, so a merely reseeded
    stream would replay the very search that already found nothing; the retry asks for the
    permuted views too, exactly as a resample does.

    The retry happens here through the max-complexity continuation (H33), which is the branch
    that looks for another problem to add once the sweep is exhausted.
    """
    domain, problem = simple_blocks
    other = a_second_problem(domain, problem)
    planner_configs = []

    def fake_compute_plans(domain_str, problem_str, planner_config):
        planner_configs.append(dict(planner_config))
        # p1 is planless; p2 gets exactly the min_number_of_plans floor, so no INC_PLANS draw
        # can keep the sweep alive past max_complexity.
        return iter([]) if "p1" in problem_str else iter([pool_plan(2)])

    monkeypatch.setattr(
        isolver,
        "_get_example_plan_computer",
        lambda config: (fake_compute_plans, dict(config["planners"]["siw"]), "siw"),
    )
    monkeypatch.setattr(isolver, "solve_step", lambda **kwargs: (Result.NO_SOLUTION, None, []))

    _, _, stats = isolver.solve_iteratively(
        domain,
        [problem, other],
        solver_config(max_complexity=3, continue_after_max_complexity=True),
    )

    assert planner_configs[0] == {"seed": 0, "restarts": 1}  # the run's own streams, untouched
    assert planner_configs[1] == {"seed": 0, "restarts": 1}
    assert planner_configs[2:] == [{"seed": 1001 * 1000003, "restarts": 2}]
    assert stats["planlessRetries"] == 1
    assert stats["deferredProblems"] == 1


def test_an_expansion_cutoff_defers_the_problem_and_retries_the_round(monkeypatch, simple_blocks):
    """(d) end to end: `solve_step` fails to build the instance for the first problem, so that
    problem is dropped from the training set and the round is retried over what is left."""
    domain, problem = simple_blocks
    other = a_second_problem(domain, problem)
    active_sets = []

    def fake_compute_plans(domain_str, problem_str, planner_config):
        return iter([pool_plan(2)])

    def fake_solve_step(**kwargs):
        names = [p.name for p in kwargs["active_problems"]]
        active_sets.append(names)
        if names == ["p1"]:
            raise ExpansionLimitExceeded("p1", 4242, 1000)
        return Result.NO_SOLUTION, None, []

    monkeypatch.setattr(
        isolver,
        "_get_example_plan_computer",
        lambda config: (fake_compute_plans, dict(config["planners"]["siw"]), "siw"),
    )
    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)

    _, _, stats = isolver.solve_iteratively(domain, [problem, other], solver_config())

    assert active_sets[0] == ["p1"]
    # The retry runs the *same* configuration over the remaining problem, not an escalated one.
    assert active_sets[1] == ["p2"]
    assert stats["expansionCutoffs"] == 1
    assert stats["deferredProblems"] == 1
