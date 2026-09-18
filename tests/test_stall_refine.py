"""Tests for H32b: a climbing sweep is not a stall.

`docs/resample-on-stall-results.md` (H32) counted every round that failed to produce (or improve)
a policy as a stalled round, including `Result.NO_SOLUTION` and the H31 frontier lower-bound
abort (`Result.FRONTIER`). On blocks3ops, once a near-general policy exists, adding a problem that
provably needs a higher feature complexity makes the sweep climb through several such abort
rounds -- each one seconds long thanks to H31 -- and the old counter mistook that climb for a
stall, firing a resample that threw away the very example plans the climb depended on.

The fix has two parts, both in `genfond/iterative_solver.py`:

1. `StallTracker.record` only increments the counter for a round that produced a policy
   (`Result.SUCCESS`, via `coverage`/`counts_toward_stall`) and did not improve the best coverage.
   `NO_SOLUTION`, `FRONTIER` (lower-bound abort included), `TIMEOUT`/`OUT_OF_RESOURCES`/`UNKNOWN`,
   and a round that itself just resampled or continued past `max_complexity` never count.
2. `StallTracker.sweep_climbing`, recomputed on every `record` call from `refuting_or_abort` (was
   this round's `Result` a refutation or a frontier lower-bound abort?) and whether there is a
   higher complexity level left (`complexity < max_complexity`). `should_resample` -- and the
   round loop's own trigger in `solve_iteratively` -- refuse to fire while it is true.

Part 1 (below "--- StallTracker: the counter ---") is tested directly against `StallTracker`,
which *is* "the trigger" the task names (`_record_round`/`_do_resample` in `iterative_solver.py`
are thin wrappers around it) -- fast, and exact about what each `Result` does to the counter.

Part 2's guard is also tested directly against `StallTracker` ("--- StallTracker: climbing ---").
Worth recording why: `sweep_climbing` is recomputed from the round that *just* ended, and only a
`Result.SUCCESS` round ever increments the counter -- every `_record_round` call site that could
push the counter up to `stall_rounds` therefore also sets `sweep_climbing = False` in that same
call, for that same round. So the *very first* round whose check would see both "the counter has
reached the threshold" and "the sweep is climbing" simultaneously does not occur in a run driven
purely by a `Result` sequence on one training set: whichever round first satisfies the counter
condition is a `SUCCESS`, and the check right after it always sees `sweep_climbing == False`. The
guard still matters -- it is what keeps a *future* design (or a path this suite does not
exercise) from resampling mid-climb -- and it is real, reachable logic, so it is tested directly
against the class it lives on rather than asserted never to fire.

The end-to-end tests (bottom) exercise the same wiring `tests/test_resample.py` and
`tests/test_no_maxc_stop.py` do: a mocked `solve_step`/`_test_policy_on_problems`, a real
`ProblemIterator`, and `simple_blocks`.
"""

from pddl.core import Plan
from pddl.logic import constants

from genfond import iterative_solver as isolver
from genfond.iterative_solver import StallTracker
from genfond.problem_iterator import ProblemIterator, Result

# --- StallTracker: the counter (H32b point 1) --------------------------------------------------


def test_the_first_success_establishes_a_baseline_without_counting_as_a_stall():
    """The very first `SUCCESS` a run sees is a strict improvement over "nothing seen yet"
    (`best_coverage_seen` starts at -1), so it resets the counter rather than incrementing it --
    matching `iterative_solver`'s only `coverage=None` default not applying here."""
    stall = StallTracker()

    stall.record(coverage=0, counts_toward_stall=True, refuting_or_abort=False, complexity=2, max_complexity=10)

    assert stall.rounds_since_improvement == 0
    assert stall.best_coverage_seen == 0
    assert stall.stall_rounds_max == 0


def test_a_non_improving_success_increments_the_counter():
    stall = StallTracker()
    stall.record(0, True, False, 2, 10)  # baseline

    stall.record(0, True, False, 2, 10)
    stall.record(0, True, False, 2, 10)

    assert stall.rounds_since_improvement == 2
    assert stall.stall_rounds_max == 2


def test_an_improving_success_resets_the_counter():
    stall = StallTracker()
    stall.record(0, True, False, 2, 10)
    stall.record(0, True, False, 2, 10)
    assert stall.rounds_since_improvement == 1

    stall.record(1, True, False, 2, 10)  # strictly more problems solved

    assert stall.rounds_since_improvement == 0
    assert stall.best_coverage_seen == 1
    # The historical peak survives a reset -- it is a running max, not the current value.
    assert stall.stall_rounds_max == 1


def test_no_solution_never_increments_the_counter():
    """(b) `Result.NO_SOLUTION` -- a genuine refutation -- is sweep progress, not a stall."""
    stall = StallTracker()
    stall.record(0, True, False, 2, 10)  # baseline

    for complexity in range(2, 9):
        stall.record(None, counts_toward_stall=False, refuting_or_abort=True, complexity=complexity, max_complexity=10)

    assert stall.rounds_since_improvement == 0
    assert stall.stall_rounds_max == 0


def test_frontier_never_increments_the_counter():
    """(b) `Result.FRONTIER`, including the H31 lower-bound abort, is the "abort" half of
    "refuting/abort" and never counts either."""
    stall = StallTracker()
    stall.record(0, True, False, 2, 10)  # baseline

    for complexity in range(2, 9):
        stall.record(None, counts_toward_stall=False, refuting_or_abort=True, complexity=complexity, max_complexity=10)

    assert stall.rounds_since_improvement == 0


def test_timeout_and_out_of_resources_never_increment_the_counter():
    """Neither refutes anything (see the `Result` docstring), so neither counts as a stall nor
    as climbing."""
    stall = StallTracker()
    stall.record(0, True, False, 2, 10)  # baseline

    stall.record(None, counts_toward_stall=False, refuting_or_abort=False, complexity=2, max_complexity=10)
    stall.record(None, counts_toward_stall=False, refuting_or_abort=False, complexity=2, max_complexity=10)

    assert stall.rounds_since_improvement == 0
    assert stall.sweep_climbing is False


def test_note_resample_resets_the_counter_but_not_best_coverage_or_the_historical_max():
    """A resample (H32's own trigger or H33's continuation) is a fresh start for the counter --
    but `best_coverage_seen` and `stall_rounds_max` are run-level bookkeeping, untouched."""
    stall = StallTracker()
    stall.record(0, True, False, 2, 10)
    stall.record(0, True, False, 2, 10)
    stall.record(0, True, False, 2, 10)
    assert stall.rounds_since_improvement == 2

    stall.note_resample()

    assert stall.rounds_since_improvement == 0
    assert stall.best_coverage_seen == 0
    assert stall.stall_rounds_max == 2


# --- StallTracker: climbing (H32b point 2) ------------------------------------------------------


def test_sweep_climbing_true_only_after_a_refuting_round_below_max_complexity():
    stall = StallTracker()

    stall.record(None, False, refuting_or_abort=True, complexity=4, max_complexity=6)

    assert stall.sweep_climbing is True


def test_sweep_climbing_false_at_the_top_of_the_sweep_even_if_refuting():
    """`complexity == max_complexity`: there is nowhere higher left to escalate to, so the round
    that follows is not "the iterator will escalate complexity next" -- it is H33's territory
    (continuing past `max_complexity`), not H32b's climbing guard."""
    stall = StallTracker()

    stall.record(None, False, refuting_or_abort=True, complexity=6, max_complexity=6)

    assert stall.sweep_climbing is False


def test_sweep_climbing_false_after_a_success_round_regardless_of_complexity():
    """A `SUCCESS` round never sets `refuting_or_abort`, so it can never leave the sweep looking
    like it is climbing, no matter how far below `max_complexity` it is."""
    stall = StallTracker()

    stall.record(0, True, refuting_or_abort=False, complexity=2, max_complexity=10)

    assert stall.sweep_climbing is False


def test_sweep_climbing_false_after_timeout_or_out_of_resources():
    """Neither is a refutation, so neither looks like climbing either, even below max_complexity."""
    stall = StallTracker()

    stall.record(None, False, refuting_or_abort=False, complexity=2, max_complexity=10)

    assert stall.sweep_climbing is False


def test_should_resample_requires_the_stall_the_budget_and_not_climbing():
    stall = StallTracker()
    stall.record(0, True, False, 2, 10)
    for _ in range(3):
        stall.record(0, True, False, 2, 10)
    assert stall.rounds_since_improvement == 3

    # Under threshold.
    assert stall.should_resample(stall_rounds=4, resamples_done=0, resample_max=1) is False
    # At threshold, budget left, not climbing: fires.
    assert stall.should_resample(stall_rounds=3, resamples_done=0, resample_max=1) is True
    # At threshold, no budget left: does not fire.
    assert stall.should_resample(stall_rounds=3, resamples_done=1, resample_max=1) is False

    # Now make the sweep look like it is climbing (a refuting round just ended, below max).
    stall.record(None, False, refuting_or_abort=True, complexity=4, max_complexity=10)
    assert stall.rounds_since_improvement == 3  # unchanged: NO_SOLUTION does not count

    # At threshold, budget left, but climbing: deferred.
    assert stall.should_resample(stall_rounds=3, resamples_done=0, resample_max=1) is False


def test_climbing_guard_defers_then_fires_once_the_sweep_stops_climbing():
    """(c) The exact H32b scenario: the counter is already at the threshold, a refuting round
    below `max_complexity` defers the resample, and once that stretch ends (here: the round that
    follows refutes the *top* of the sweep, so there is nowhere left to climb to) the very next
    check fires."""
    stall = StallTracker()
    stall.record(0, True, False, 2, 6)
    for _ in range(3):
        stall.record(0, True, False, 2, 6)  # counter: 1, 2, 3
    assert stall.rounds_since_improvement == 3

    # Complexity climbs via genuine refutations; the counter does not move.
    stall.record(None, False, True, complexity=3, max_complexity=6)
    assert stall.should_resample(3, resamples_done=0, resample_max=2) is False  # deferred: climbing
    stall.record(None, False, True, complexity=4, max_complexity=6)
    assert stall.should_resample(3, resamples_done=0, resample_max=2) is False  # still deferred
    assert stall.deferred == 0  # StallTracker itself does not count deferrals -- see below

    # The sweep reaches the top: nothing higher left to escalate to, so this is no longer
    # "climbing" by the guard's own definition (H33's territory starts here, not H32b's).
    stall.record(None, False, True, complexity=6, max_complexity=6)

    assert stall.sweep_climbing is False
    assert stall.should_resample(3, resamples_done=0, resample_max=2) is True


# --- end to end through solve_iteratively -------------------------------------------------------


class DummyPolicy:
    def __init__(self, cost=(1000,)):
        self.cost = cost


A, B, C = constants("a b c")
CYCLE = [("pick", [A, B]), ("put", [A, C]), ("pick", [A, C]), ("put", [A, B])]


def pool_plan(n):
    """A genuinely executable plan for `simple_blocks` -- `PlanStateCoverage`/`StateSpaceGraph`
    replay every example plan for real."""
    return Plan([CYCLE[i % len(CYCLE)] for i in range(1, n + 1)])


def solver_config(**overrides):
    config = {
        "min_complexity": 2,
        "max_complexity": 40,
        "use_unrestricted_features": False,
        "reset_complexity_on_state_space_change": False,
        "resample_reset_complexity": False,
        "add_problem_after_success": False,
        "unselect_problems": False,
        "min_number_of_plans": 2,
        "max_plans_per_problem": None,
        "max_frontier_expansions": 20,
        "max_frontier_states_per_round": 0,
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
        "stall_rounds": 6,
        "resample_max": 1,
        "continue_after_max_complexity": True,
    }
    config.update(overrides)
    return config


def run_rounds(monkeypatch, simple_blocks, results, **config_overrides):
    """Run `solve_iteratively` for exactly `len(results)` rounds, each returning the given
    `Result` in order -- a `SUCCESS` gets a fresh, never-improving `DummyPolicy` (validated
    coverage is always 0/1: `_test_policy_on_problems` is mocked to report nothing solved, so a
    run of `SUCCESS`es alone never improves). Returns `(stats, iterator)`.
    """
    domain, problem = simple_blocks
    planner_configs: list[dict] = []

    def fake_compute_plans(domain_str, problem_str, planner_config):
        index = len(planner_configs)
        planner_configs.append(dict(planner_config))
        return iter([pool_plan(2 * index + 1), pool_plan(2 * index + 2)])

    calls = {"n": 0}

    def fake_solve_step(**kwargs):
        result = results[calls["n"]]
        calls["n"] += 1
        policy = DummyPolicy() if result == Result.SUCCESS else None
        return (result, policy, [])

    monkeypatch.setattr(isolver, "_test_policy_on_problems", lambda *a, **k: [])
    monkeypatch.setattr(
        isolver,
        "_get_example_plan_computer",
        lambda config: (fake_compute_plans, dict(config["planners"]["siw"]), "siw"),
    )
    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)

    iterators: list[ProblemIterator] = []

    class CapturingProblemIterator(ProblemIterator):
        def __iter__(self):
            iterators.append(self)
            return super().__iter__()

        def __next__(self):
            if calls["n"] >= len(results):
                raise StopIteration
            return super().__next__()

    monkeypatch.setattr(isolver, "ProblemIterator", CapturingProblemIterator)

    _, _, stats = isolver.solve_iteratively(domain, [problem], solver_config(**config_overrides))
    return stats, iterators[0]


def test_end_to_end_six_non_improving_success_rounds_trigger_a_resample(monkeypatch, simple_blocks):
    """(a) Round 1 establishes the 0/1 baseline; rounds 2-7 are six non-improving `SUCCESS`es,
    reaching the `stall_rounds: 6` threshold; the resample fires at the top of round 8, before
    its own `solve_step` call."""
    stats_short, _ = run_rounds(monkeypatch, simple_blocks, [Result.SUCCESS] * 7)
    assert "resamples" not in stats_short
    assert stats_short["stallRoundsMax"] == 6

    stats_long, iterator = run_rounds(monkeypatch, simple_blocks, [Result.SUCCESS] * 8)
    assert stats_long["resamples"] == 1
    assert stats_long["resampledPlans"] == 2  # the min_number_of_plans floor, redrawn
    assert "stallDeferred" not in stats_long


def test_end_to_end_abort_and_no_solution_rounds_never_trigger_a_resample(monkeypatch, simple_blocks):
    """(b) A long run of genuine refutations and frontier lower-bound aborts alone: with
    `stall_rounds: 1` the old counter would have resampled on round 2 already; H32b's fix means
    it never does, because neither `Result` ever counts toward the counter."""
    results = [Result.NO_SOLUTION, Result.FRONTIER, Result.TIMEOUT, Result.OUT_OF_RESOURCES] * 4
    stats, _ = run_rounds(monkeypatch, simple_blocks, results, stall_rounds=1, resample_max=5)

    assert "resamples" not in stats
    assert stats["stallRoundsMax"] == 0


def test_end_to_end_no_double_resample_in_the_round_the_continuation_fires(monkeypatch, simple_blocks):
    """(d) A pure `NO_SOLUTION` climb to `max_complexity` (as in `test_no_maxc_stop.py`) with an
    aggressive `stall_rounds` that would, under the *old* counting, have also made H32's own
    trigger stall-ready well before the top of the sweep. Under H32b it never gets the chance:
    `NO_SOLUTION` never increments the counter, so only H33's continuation ever resamples here --
    exactly one resample for the round it fires in, not two."""
    # Round 1: complexity 2. Round 2: complexity 3 == max_complexity, still NO_SOLUTION -- the
    # sweep is exhausted. Round 3's own __next__() call is where H33's continuation resamples and
    # restarts the sweep; a third result lets that round actually run.
    results = [Result.NO_SOLUTION] * 3
    stats, iterator = run_rounds(
        monkeypatch,
        simple_blocks,
        results,
        max_complexity=3,
        stall_rounds=1,  # would have been stall-ready every round under the old counting
        resample_max=1,
    )

    assert stats["resamples"] == 1
    assert stats["maxComplexityContinuations"] == 1
    assert stats["stallRoundsMax"] == 0  # NO_SOLUTION never counted, so the counter never moved
    assert "stallDeferred" not in stats
