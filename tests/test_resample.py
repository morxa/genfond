"""Tests for resampling the example plans on a stall (H32).

Which policy the loop converges to is decided by the example plans it happens to have sampled: on
blocks3ops with `--type datalog-sig`, 2 of 6 seeds settle within a handful of rounds on a
patchwork policy whose coverage then never improves again, while the other 4 reach 95/95. Nothing
in the escalation ladder can undo an unlucky sample -- it only ever adds plans to it -- so when
the best coverage stalls, the planner's plans are thrown away and drawn again from a stream with
a different RNG state.

The end-to-end tests mock `solve_step` and `execute_policy` (as `tests/test_policy_plans.py` and
`tests/test_in_loop_validation.py` do) but run the real `ProblemIterator`, so the plan
bookkeeping is genuinely exercised.
"""

import random

from pddl.core import Plan
from pddl.logic import Constant, constants

from genfond import iterative_solver as isolver
from genfond.problem_iterator import PlanStateCoverage, ProblemIterator, Result, plan_key

# --- ProblemIterator-level unit tests ---------------------------------------------------------


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
        "min_number_of_plans": 3,
        "max_frontier_expansions": 20,
        "reset_complexity_on_state_space_change": False,
        "resample_reset_complexity": False,
        "add_problem_after_success": False,
    }
    config.update(overrides)
    return config


def a_started_iterator(config, num_plans=20):
    """An iterator over two problems, each with a long stream of distinguishable plans."""
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    plans = {p.name: iter([a_plan(f"{p.name}-{i}") for i in range(num_plans)]) for p in problems}
    iterator = iter(ProblemIterator(problems, config, plans=plans))
    return iterator, next(iterator)


def test_resample_replaces_only_planner_plans():
    """(b) The policy-conformant plans of H27 and the policy-prefix plans of H30 sit in front of
    a problem's list and are counted in `policy_plan_counts`; they carry information about the
    best policy so far and must survive a resample untouched. Everything behind them is the
    sample, and is replaced one for one."""
    iterator, _ = a_started_iterator(iterator_config())
    # Three planner plans (the min_number_of_plans floor) plus one policy-conformant plan.
    assert len(iterator.active_plans["p1"]) == 3
    iterator.record_policy_plans({"p1": a_plan("policy-step")})
    before = list(iterator.active_plans["p1"])
    assert iterator.policy_plan_counts["p1"] == 1

    replaced, affected = iterator.resample_planner_plans()

    assert (replaced, affected) == (3, 1)
    after = iterator.active_plans["p1"]
    # Same size, same policy plan in front, and not one of the planner plans is still there.
    assert len(after) == len(before)
    assert plan_key(after[0]) == plan_key(a_plan("policy-step"))
    assert iterator.policy_plan_counts["p1"] == 1
    assert {plan_key(p) for p in after[1:]}.isdisjoint({plan_key(p) for p in before[1:]})
    # The replacements come from the problem's own stream, continued past what it had drawn.
    assert [str(p.actions[0][0]) for p in after[1:]] == ["p1-3", "p1-4", "p1-5"]


def test_resample_rebuilds_the_state_coverage_tracker():
    """`PlanStateCoverage` only ever grows, which is right while plans are only ever added. A
    resample *removes* plans, so the states only those reached must stop counting as covered --
    otherwise every replacement plan is rejected as redundant."""
    coverage = PlanStateCoverage.__new__(PlanStateCoverage)
    coverage.covered = {"p1": {"s1", "s2"}}
    coverage.reset("p1")
    assert "p1" not in coverage.covered
    # A reset of a problem that was never tracked is a no-op, not a KeyError.
    coverage.reset("p2")


def test_resample_invalidates_the_refutation_and_leaves_max_cost_alone():
    """(c) The plan set changed, so no complexity level is refuted any more -- exactly as
    INC_PLANS and `record_policy_plans` drop the bound. `max_cost` is deliberately kept: a
    resample is not permission to accept a policy worse than the one already found."""
    iterator, _ = a_started_iterator(iterator_config())
    iterator.set_last_result(Result.NO_SOLUTION)
    next(iterator)  # INC_COMPLEXITY to 3
    iterator.set_last_result(Result.SUCCESS, cost=(6,))
    assert iterator.refuted_complexity == 3
    assert iterator.enforce_highest_complexity() is True
    assert iterator.max_cost == 5
    complexity_before = iterator.complexity

    replaced, _ = iterator.resample_planner_plans()

    assert replaced
    assert iterator.refuted_complexity == iterator.config["min_complexity"] - 1
    assert iterator.enforce_highest_complexity() is False
    assert iterator.max_cost == 5
    # No problem was added and the sweep stayed where it was.
    assert [p.name for p in iterator.active_problems] == ["p1"]
    assert iterator.complexity == complexity_before
    # The bound the add-a-problem branch would otherwise carry over is dropped too: enlarging an
    # instance's state space is not the monotone step that argument covers.
    assert iterator.plans_added_since_success is True


def test_resample_reset_complexity_restarts_the_sweep():
    """(4) The optional switch, mirroring reset_complexity_on_state_space_change."""
    iterator, _ = a_started_iterator(iterator_config(resample_reset_complexity=True))
    iterator.set_last_result(Result.NO_SOLUTION)
    next(iterator)
    assert iterator.complexity == 3

    iterator.resample_planner_plans()

    assert iterator.complexity == iterator.config["min_complexity"]
    assert iterator.sweep_target == 3


def test_resample_falls_back_to_a_fresh_stream_when_the_problem_s_own_runs_dry():
    """The problem's own stream is continued first -- the planner yields lazily and dedupes
    within a stream, so its next plans are new ones that cost nothing extra to reach. Only when
    it is exhausted is a freshly seeded stream asked for, and it then becomes the problem's
    stream so later INC_PLANS draws continue there."""
    # Exactly the three plans the floor consumes, so the stream is empty at resample time.
    iterator, _ = a_started_iterator(iterator_config(), num_plans=3)
    calls = []

    def fresh(problem):
        calls.append(problem.name)
        return iter([a_plan(f"fresh-{problem.name}-{i}") for i in range(5)])

    replaced, affected = iterator.resample_planner_plans(fresh)

    assert (replaced, affected) == (3, 1)
    assert calls == ["p1"]
    assert [str(p.actions[0][0]) for p in iterator.active_plans["p1"]] == [
        "fresh-p1-0",
        "fresh-p1-1",
        "fresh-p1-2",
    ]
    # The remaining two plans of the fresh stream are what the next INC_PLANS draws.
    assert str(next(iterator.plan_iterators["p1"]).actions[0][0]) == "fresh-p1-3"


def test_resample_keeps_the_existing_plans_when_nothing_new_can_be_drawn():
    """A stream with nothing left and no fresh one to fall back on leaves the problem exactly as
    it was -- and, since nothing changed, leaves the refutation bookkeeping alone too."""
    iterator, _ = a_started_iterator(iterator_config(), num_plans=3)
    iterator.set_last_result(Result.NO_SOLUTION)
    next(iterator)
    iterator.set_last_result(Result.NO_SOLUTION)
    assert iterator.refuted_complexity == 3
    before = list(iterator.active_plans["p1"])

    assert iterator.resample_planner_plans() == (0, 0)

    assert iterator.active_plans["p1"] == before
    assert iterator.refuted_complexity == 3


def test_resample_does_nothing_for_a_problem_without_planner_plans():
    """A problem holding only policy/prefix plans has no sample to redraw."""
    config = iterator_config(min_number_of_plans=0)
    iterator, _ = a_started_iterator(config)
    assert iterator.active_plans["p1"] == []
    iterator.record_policy_plans({"p1": a_plan("policy-step")})

    assert iterator.resample_planner_plans() == (0, 0)
    assert [plan_key(p) for p in iterator.active_plans["p1"]] == [plan_key(a_plan("policy-step"))]


# --- end to end through solve_iteratively -----------------------------------------------------


class DummyPolicy:
    def __init__(self, cost=(2,)):
        self.cost = cost


# Real, applicable plans for the `simple_blocks` fixture: `PlanStateCoverage` replays every
# example plan against the actual domain, so a made-up action name would not survive the trip.
# Plan n is the first n actions of the pick/put cycle, so all of them are distinct.
A, B, C = constants("a b c")
CYCLE = [("pick", [A, B]), ("put", [A, C]), ("pick", [A, C]), ("put", [A, B])]


def pool_plan(n):
    return Plan([CYCLE[i % len(CYCLE)] for i in range(1, n + 1)])


def solver_config(**overrides):
    config = {
        "min_complexity": 2,
        # High enough that the INC_COMPLEXITY branch keeps the ladder alive for every round these
        # tests need; the ladder raises StopIteration once it runs out of escalations.
        "max_complexity": 40,
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
        "stall_rounds": 3,
        "resample_max": 1,
    }
    config.update(overrides)
    return config


def run_stalling_rounds(monkeypatch, simple_blocks, num_rounds, **config_overrides):
    """`num_rounds` rounds that never produce a policy, so the best coverage never improves.

    Every call to the planner hands out a fresh two-plan stream from a disjoint slice of the
    plan pool, so which stream a problem's plans came from is visible in the plans themselves.
    Two plans per stream is exactly the `min_number_of_plans` floor, so a stream is always
    exhausted by the time a resample wants more -- which is what makes the fresh-stream fallback
    (and with it the reseeded planner config) part of what these tests exercise.

    Returns `(iterator, stats, planner_configs)`.
    """
    domain, problem = simple_blocks
    planner_configs: list[dict] = []

    def fake_compute_plans(domain_str, problem_str, planner_config):
        index = len(planner_configs)
        planner_configs.append(dict(planner_config))
        return iter([pool_plan(2 * index + 1), pool_plan(2 * index + 2)])

    monkeypatch.setattr(
        isolver,
        "_get_example_plan_computer",
        lambda config: (fake_compute_plans, dict(config["planners"]["siw"]), "siw"),
    )
    monkeypatch.setattr(isolver, "solve_step", lambda **kwargs: (Result.NO_SOLUTION, None, []))

    iterators: list[ProblemIterator] = []
    rounds = {"n": 0}

    class CapturingProblemIterator(ProblemIterator):
        def __iter__(self):
            iterators.append(self)
            return super().__iter__()

        def __next__(self):
            rounds["n"] += 1
            if rounds["n"] > num_rounds:
                raise StopIteration
            return super().__next__()

    monkeypatch.setattr(isolver, "ProblemIterator", CapturingProblemIterator)

    _, _, stats = isolver.solve_iteratively(domain, [problem], solver_config(**config_overrides))
    return iterators[0], stats, planner_configs


def test_the_stall_counter_triggers_after_stall_rounds_and_not_before(monkeypatch, simple_blocks):
    """(a) With `stall_rounds: 3`, three non-improving rounds must not resample; the fourth round
    -- the one that starts with the counter already at 3 -- must."""
    _, stats_short, _ = run_stalling_rounds(monkeypatch, simple_blocks, num_rounds=3)
    assert "resamples" not in stats_short
    assert stats_short["stallRoundsMax"] == 3

    iterator, stats_long, configs = run_stalling_rounds(monkeypatch, simple_blocks, num_rounds=4)
    assert stats_long["resamples"] == 1
    assert stats_long["resampledPlans"] == 2  # the min_number_of_plans floor, redrawn
    # The problem now holds the second stream's plans instead of the first stream's.
    assert [plan_key(p) for p in iterator.active_plans["p1"]] == [plan_key(pool_plan(3)), plan_key(pool_plan(4))]
    assert len(configs) == 2


def test_the_fresh_stream_is_reseeded_and_asks_for_at_least_two_restarts(monkeypatch, simple_blocks):
    """SIW's restart 1 is the identity view of the task, so the seed only reaches the permuted
    views from restart 2 on: at the measured `restarts: 1` a merely reseeded stream would replay
    the very same plans. The resample therefore changes both."""
    _, _, configs = run_stalling_rounds(monkeypatch, simple_blocks, num_rounds=4)

    assert configs[0] == {"seed": 0, "restarts": 1}  # the run's own stream, untouched
    assert configs[1] == {"seed": 1000003, "restarts": 2}


def test_resample_max_is_respected(monkeypatch, simple_blocks):
    """(d) Enough stalled rounds for three resamples still yield exactly `resample_max`."""
    _, stats, configs = run_stalling_rounds(monkeypatch, simple_blocks, num_rounds=16, resample_max=1)
    assert stats["resamples"] == 1
    assert [c["seed"] for c in configs] == [0, 1000003]

    _, stats, configs = run_stalling_rounds(monkeypatch, simple_blocks, num_rounds=16, resample_max=2)
    assert stats["resamples"] == 2
    assert [c["seed"] for c in configs] == [0, 1000003, 2000006]


def test_resample_off_does_nothing(monkeypatch, simple_blocks):
    """(e) The switch off leaves the run exactly as it was before this mechanism existed."""
    iterator, stats, configs = run_stalling_rounds(monkeypatch, simple_blocks, num_rounds=16, resample_on_stall=False)
    assert "resamples" not in stats
    assert "resampledPlans" not in stats
    assert len(configs) == 1
    assert [plan_key(p) for p in iterator.active_plans["p1"]] == [plan_key(pool_plan(1)), plan_key(pool_plan(2))]


def test_resample_is_inert_without_example_plans(monkeypatch, simple_blocks):
    """The default is true, so it must be a no-op for the rule-based types, whose state space is
    not plan-restricted and which have no planner at all."""
    _, stats, configs = run_stalling_rounds(monkeypatch, simple_blocks, num_rounds=16, use_example_plans=False)
    assert "resamples" not in stats
    assert configs == []


def test_stall_rounds_or_resample_max_of_zero_switches_the_mechanism_off(monkeypatch, simple_blocks):
    _, stats, _ = run_stalling_rounds(monkeypatch, simple_blocks, num_rounds=16, stall_rounds=0)
    assert "resamples" not in stats
    _, stats, _ = run_stalling_rounds(monkeypatch, simple_blocks, num_rounds=16, resample_max=0)
    assert "resamples" not in stats


def test_the_global_rng_state_survives_a_resample(monkeypatch, simple_blocks):
    """Policy execution draws on the global RNG, so a resample that left it advanced would change
    the validation results of every later round and break `--seed` reproducibility."""
    random.seed(1234)
    expected = [random.random() for _ in range(3)]

    random.seed(1234)
    _, stats, _ = run_stalling_rounds(monkeypatch, simple_blocks, num_rounds=16, resample_max=2)
    assert stats["resamples"] == 2
    assert [random.random() for _ in range(3)] == expected
