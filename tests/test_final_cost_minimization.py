"""Unit tests for the final cost minimization pass (`add_problem_after_success` +
`final_cost_minimization`).

These mock out `solve_step` and `execute_policy` rather than driving a real domain through
clingo, so they exercise the pass's own control flow (which complexities it tries, when it
stops, which candidate it keeps) in isolation -- see `docs/final-climb-results.md` for the
PDDL-level validation runs on the workstation.
"""

import types

from genfond import iterative_solver as isolver
from genfond.iterative_solver import _final_cost_minimization_pass, _test_policy_on_problems
from genfond.problem_iterator import Result
from genfond.shutdown import request_stop, reset_stop


class DummyProblem:
    def __init__(self, name, objects=()):
        self.name = name
        self.objects = objects


class DummyPolicy:
    def __init__(self, cost):
        self.cost = cost


def make_problem_iterator(active_problems, succ_complexity=2, solved=None):
    if solved is None:
        solved = {p.name: True for p in active_problems}
    return types.SimpleNamespace(
        active_problems=active_problems,
        active_plans={},
        dead_states={},
        succ_complexity=succ_complexity,
        solved=solved,
    )


def base_config(**overrides):
    config = {
        "max_complexity": 6,
        "use_unrestricted_features": False,
        "policy_iterations": 1,
        "validation_iterations": 1,
        "minimize_good_signatures": "none",
        "minimize_selected_count": "none",
    }
    config.update(overrides)
    return config


def test_test_policy_on_problems_tests_every_problem_even_after_a_failure(monkeypatch):
    """Unlike the main loop's testing block, this must not stop at the first miss: the final
    pass compares candidates by how many problems they solve overall, so an early break would
    undercount whichever candidate happens to fail on a smaller problem."""
    problems = [DummyProblem("p1"), DummyProblem("p2"), DummyProblem("p3")]
    calls = []

    def fake_execute_policy(domain, problem, policy, config, **kwargs):
        calls.append(problem.name)
        if problem.name == "p1":
            raise RuntimeError("no solution")
        return []

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    solved = _test_policy_on_problems(None, problems, DummyPolicy((1,)), base_config())
    assert [p.name for p in solved] == ["p2", "p3"]
    assert calls == ["p1", "p2", "p3"]


def test_final_pass_climbs_through_no_solution_and_keeps_cheapest_valid_candidate(monkeypatch):
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=6)
    stats = {}
    policy = DummyPolicy((5,))  # starting cost 5, at complexity 2

    calls = []

    def fake_solve_step(**kwargs):
        calls.append(kwargs["complexity"])
        if kwargs["complexity"] == 3:
            # Nothing beats max_cost=4 at complexity 3; the pass must keep climbing exactly
            # like the INC_COMPLEXITY branch does on NO_SOLUTION.
            return Result.NO_SOLUTION, None, []
        if kwargs["complexity"] == 4:
            return Result.SUCCESS, DummyPolicy((2,)), []
        raise AssertionError(f"unexpected complexity {kwargs['complexity']}")

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config, **kwargs: [])

    best_policy, best_solved = _final_cost_minimization_pass(None, problems, problem_iterator, config, stats, policy)

    # complexity 3 (NO_SOLUTION, keeps climbing) then complexity 4 (SUCCESS, cost 2).
    # max_cost after that success is 1, which is not > complexity 4, so the loop stops there.
    assert calls == [3, 4]
    assert best_policy.cost == (2,)
    assert [p.name for p in best_solved] == ["p1", "p2"]
    assert stats["finalPassRounds"] == 2
    assert stats["finalPassCostBefore"] == 5
    assert stats["finalPassCostAfter"] == 2


def test_final_pass_discards_a_candidate_that_fails_training_set_execution(monkeypatch):
    """A round's ASP-level SUCCESS only proves the model satisfies the constraints; execution
    can still fail (e.g. a cycle), the same distinction the main loop's own testing makes."""
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=6)
    stats = {}
    policy = DummyPolicy((5,))

    def fake_solve_step(**kwargs):
        return Result.SUCCESS, DummyPolicy((1,)), []

    def fake_execute_policy(domain, problem, policy, config, **kwargs):
        if problem.name == "p2":
            raise RuntimeError("cycle")
        return []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    best_policy, best_solved = _final_cost_minimization_pass(None, problems, problem_iterator, config, stats, policy)

    # The complexity-3 candidate (cost 1) does not solve p2, so it must be discarded even
    # though it is cheaper; the original policy (cost 5) is kept.
    assert best_policy is policy
    assert stats["finalPassCostAfter"] == 5


def test_final_pass_stops_after_final_pass_max_levels_even_when_max_cost_allows_more(monkeypatch):
    """The Observation this branch fixes: on the cluster the pass climbed unbounded (NO_SOLUTION
    never tightens max_cost, so the old `while max_cost > complexity` condition alone never
    stops it) until a single solve_step call exceeded any budget. final_pass_max_levels must cap
    the number of complexity levels tried regardless of max_cost/max_complexity."""
    problems = [DummyProblem("p1")]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=100, final_pass_max_levels=2)
    stats = {}
    policy = DummyPolicy((50,))  # max_cost starts at 49, far above max_complexity=100 would allow

    calls = []

    def fake_solve_step(**kwargs):
        calls.append(kwargs["complexity"])
        # Never tightens max_cost, so without the level cap this would climb all the way to
        # max_complexity=100.
        return Result.NO_SOLUTION, None, []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config, **kwargs: [])

    best_policy, best_solved = _final_cost_minimization_pass(None, problems, problem_iterator, config, stats, policy)

    assert calls == [3, 4]  # exactly final_pass_max_levels=2 rounds, not climbing to 100
    assert best_policy is policy
    assert stats["finalPassRounds"] == 2
    assert stats["finalPassLevelsTried"] == 2


def test_final_pass_max_levels_null_keeps_the_old_unbounded_climb(monkeypatch):
    """final_pass_max_levels: null (the default `config.get` sees when the key is absent, and
    what a config file can set explicitly) must reproduce the previous unbounded pass -- the
    climb only stops via max_cost/max_complexity, as covered by the pre-existing tests above,
    but this exercises it past what the new default (2) would allow, to prove null really lifts
    the cap rather than silently applying some other default."""
    problems = [DummyProblem("p1")]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=6, final_pass_max_levels=None)
    stats = {}
    policy = DummyPolicy((50,))  # max_cost stays 49 throughout (NO_SOLUTION never tightens it)

    calls = []

    def fake_solve_step(**kwargs):
        calls.append(kwargs["complexity"])
        return Result.NO_SOLUTION, None, []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config, **kwargs: [])

    _final_cost_minimization_pass(None, problems, problem_iterator, config, stats, policy)

    # Climbs every level up to max_complexity=6 (4 rounds: complexity 3, 4, 5, 6), well past the
    # new default cap of 2 -- proving null disables final_pass_max_levels entirely.
    assert calls == [3, 4, 5, 6]
    assert stats["finalPassRounds"] == 4
    assert stats["finalPassLevelsTried"] == 4


def test_final_pass_threads_final_pass_max_pool_into_solve_step(monkeypatch):
    """final_pass_max_pool is config-only (no CLI flag) and must reach solve_step's
    max_pool_size kwarg unchanged -- the actual pool-size check lives in solve() (see its
    docstring/comment), which is out of scope for these solve_step-mocking tests."""
    problems = [DummyProblem("p1")]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=6, final_pass_max_pool=500)
    stats = {}
    policy = DummyPolicy((5,))

    seen_pool_sizes = []

    def fake_solve_step(**kwargs):
        seen_pool_sizes.append(kwargs["max_pool_size"])
        return Result.NO_SOLUTION, None, []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config, **kwargs: [])

    _final_cost_minimization_pass(None, problems, problem_iterator, config, stats, policy)

    assert seen_pool_sizes == [500, 500]


def test_final_pass_stops_gracefully_on_out_of_resources(monkeypatch):
    problems = [DummyProblem("p1")]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=6)
    stats = {}
    policy = DummyPolicy((5,))

    calls = []

    def fake_solve_step(**kwargs):
        calls.append(kwargs["complexity"])
        return Result.OUT_OF_RESOURCES, None, []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config, **kwargs: [])

    best_policy, best_solved = _final_cost_minimization_pass(None, problems, problem_iterator, config, stats, policy)

    assert calls == [3]
    assert best_policy is policy
    assert stats["finalPassRounds"] == 1
    assert stats["finalPassCostAfter"] == 5


def solve_iteratively_config(**overrides):
    """A minimal config for driving `solve_iteratively` end to end with `solve_step` and
    `execute_policy` mocked out -- enough keys for `ProblemIterator` and the surrounding
    function, none of the feature-generation/clingo machinery underneath `solve_step`."""
    config = {
        "min_complexity": 2,
        "max_complexity": 6,
        "use_unrestricted_features": False,
        "reset_complexity_on_state_space_change": False,
        "add_problem_after_success": True,
        "unselect_problems": False,
        "min_number_of_plans": 5,
        "max_frontier_expansions": 20,
        "use_example_plans": False,
        "frontier_expansion": False,
        "policy_iterations": 1,
        "validation_iterations": 1,
        "stop_after_first_solution": True,
        "final_cost_minimization": True,
        "minimize_good_signatures": "none",
        "minimize_selected_count": "none",
    }
    config.update(overrides)
    return config


def test_final_pass_runs_when_the_iterator_exhausts_without_solving_everything(monkeypatch):
    """The trigger the coordinator asked for: on a suite where not every problem gets solved,
    `stop_after_first_solution`'s break never fires, so the main `for` loop can only end by the
    `ProblemIterator` raising `StopIteration` (here: a lone success is followed by NO_SOLUTION,
    then OUT_OF_RESOURCES, which closes off every escalation branch and ends the loop). The
    final pass must still run once a policy exists -- it is called unconditionally after the
    loop, not only from inside the `stop_after_first_solution` break, so this must hold without
    any special-casing for *why* the loop ended (StopIteration vs. break), as long as it wasn't
    an unhandled exception (see the companion exception test below)."""
    p1 = DummyProblem("p1", objects=[1])
    p2 = DummyProblem("p2", objects=[1, 2])
    problems = [p1, p2]
    config = solve_iteratively_config()

    solve_step_script = [
        (Result.SUCCESS, DummyPolicy((2,)), []),  # round 1: p1 alone solves at complexity 2
        (Result.NO_SOLUTION, None, []),  # round 2: p1+p2 together, complexity 2 -- fails
        (Result.OUT_OF_RESOURCES, None, []),  # round 3: complexity 3 -- exhausts the ladder
    ]

    def fake_solve_step(**kwargs):
        return solve_step_script.pop(0)

    def fake_execute_policy(domain, problem, policy, config, **kwargs):
        if problem.name == "p2":
            raise RuntimeError("no action found")
        return []

    final_pass_calls = []

    def fake_final_pass(domain, problems, problem_iterator, config, stats, policy, wall_deadline=None):
        final_pass_calls.append((problem_iterator.active_problems, policy))
        return DummyPolicy((1,)), [p1, p2]

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    monkeypatch.setattr(isolver, "_final_cost_minimization_pass", fake_final_pass)

    policy, solved, stats = isolver.solve_iteratively(None, problems, config)

    assert solve_step_script == []  # every scripted round was consumed; the loop really ran
    # to StopIteration, not stopped short by a bug elsewhere.
    assert len(final_pass_calls) == 1
    active_problems, passed_policy = final_pass_calls[0]
    assert {p.name for p in active_problems} == {"p1", "p2"}
    assert passed_policy.cost == (2,)  # the round-1 policy, since rounds 2/3 never succeeded
    # solve_iteratively must return whatever the (mocked) final pass decided, not the
    # pre-pass policy/solved set.
    assert policy.cost == (1,)
    assert solved == [p1, p2]


def test_final_pass_does_not_run_after_an_unhandled_exception(monkeypatch):
    """An exception that escapes `solve_step` must abort the whole run, not be swallowed into a
    (spurious) final pass over whatever policy happened to exist beforehand."""
    p1 = DummyProblem("p1", objects=[1])
    problems = [p1]
    config = solve_iteratively_config()

    def fake_solve_step(**kwargs):
        raise ValueError("boom")

    final_pass_calls = []
    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(
        isolver, "_final_cost_minimization_pass", lambda *a, **k: final_pass_calls.append(1) or (None, [])
    )

    try:
        isolver.solve_iteratively(None, problems, config)
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert final_pass_calls == []


# --- wall-clock deadline handling (hyp/final-pass-deadline) --------------------------------
#
# A cluster run with `--max-wall-time 13200` (reserve 300s) and `final_cost_minimization: true`
# was killed by SLURM instead of stopping gracefully: the main loop stopped at the deadline, but
# the pass then ran its own climb rounds with nothing checking the deadline, well past the
# reserve. These tests cover the fix: the pass now respects `wall_deadline` the same way the
# main loop does (skip outright if there isn't enough time left, stop between rounds if the
# deadline passes while it's running), and the main loop itself stops `final_pass_budget`
# seconds earlier than before when the pass is enabled, so the pass has a slice of its own
# instead of whatever happens to be left over.


def test_final_pass_skipped_when_the_deadline_is_already_exhausted(monkeypatch):
    """`wall_deadline` already in the past (however the main loop ended) must skip the pass
    outright -- no `solve_step` call at all -- and report why via `finalPassSkipped`."""
    p1, p2 = DummyProblem("p1"), DummyProblem("p2")
    problems = [p1, p2]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2, solved={"p1": True, "p2": False})
    config = base_config(max_complexity=6)
    stats = {}
    policy = DummyPolicy((5,))

    def fake_solve_step(**kwargs):
        raise AssertionError("solve_step must not be called once the deadline has passed")

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)

    wall_deadline = isolver.time.perf_counter() - 10.0
    best_policy, best_solved = _final_cost_minimization_pass(
        None, problems, problem_iterator, config, stats, policy, wall_deadline=wall_deadline
    )

    assert best_policy is policy
    # Falls back to what problem_iterator already knows was solved, without re-executing.
    assert [p.name for p in best_solved] == ["p1"]
    assert stats["finalPassSkipped"] == "deadline_exhausted"
    assert "finalPassRounds" not in stats


def test_final_pass_skipped_when_a_stop_is_pending(monkeypatch):
    """SIGINT/SIGTERM (genfond.shutdown.stop_requested) must skip the pass outright too, exactly
    like it stops the main loop from starting another round."""
    p1 = DummyProblem("p1")
    problems = [p1]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=6)
    stats = {}
    policy = DummyPolicy((5,))

    def fake_solve_step(**kwargs):
        raise AssertionError("solve_step must not be called while a stop is pending")

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)

    try:
        request_stop()
        best_policy, best_solved = _final_cost_minimization_pass(
            None, problems, problem_iterator, config, stats, policy, wall_deadline=isolver.time.perf_counter() + 3600
        )
    finally:
        reset_stop()

    assert best_policy is policy
    assert stats["finalPassSkipped"] == "stop_requested"


def test_final_pass_skipped_when_wall_time_stop_leaves_too_little_time(monkeypatch):
    """When the main loop stopped because its own wall budget ran out (`stoppedBy ==
    "wall_time"`), the pass only starts if `final_pass_min_time` seconds remain before its
    deadline -- otherwise it would just start a round it likely has to cut off immediately."""
    p1 = DummyProblem("p1")
    problems = [p1]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=6, final_pass_min_time=600)
    stats = {"stoppedBy": "wall_time"}
    policy = DummyPolicy((5,))

    def fake_solve_step(**kwargs):
        raise AssertionError("solve_step must not be called with insufficient remaining time")

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)

    # Only 5s remain before the deadline, far short of final_pass_min_time=600.
    wall_deadline = isolver.time.perf_counter() + 5.0
    best_policy, best_solved = _final_cost_minimization_pass(
        None, problems, problem_iterator, config, stats, policy, wall_deadline=wall_deadline
    )

    assert best_policy is policy
    assert stats["finalPassSkipped"] == "insufficient_wall_time"


def test_final_pass_min_time_gate_is_not_applied_when_not_stopped_by_wall_time(monkeypatch):
    """The `final_pass_min_time` gate only fires when the main loop stopped *because of* the
    wall budget. If it stopped for another reason (e.g. every problem solved), even a sliver of
    remaining time is worth attempting a round -- the pass must not be skipped outright."""
    p1 = DummyProblem("p1")
    problems = [p1]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=6, final_pass_min_time=600)
    stats = {}  # no "stoppedBy" at all -- the stop_after_first_solution break path
    policy = DummyPolicy((5,))

    calls = []

    def fake_solve_step(**kwargs):
        calls.append(kwargs["complexity"])
        return Result.NO_SOLUTION, None, []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config, **kwargs: [])

    wall_deadline = isolver.time.perf_counter() + 5.0
    _final_cost_minimization_pass(None, problems, problem_iterator, config, stats, policy, wall_deadline=wall_deadline)

    # Climbs complexity 3, then 4 (still NO_SOLUTION, max_cost stays 4); stops there because
    # max_cost(4) > complexity(4) is false -- an ordinary, non-deadline-related loop exit.
    assert calls == [3, 4]
    assert "finalPassSkipped" not in stats


def test_final_pass_stops_between_rounds_when_the_deadline_passes(monkeypatch):
    """A pass already running must stop before starting its *next* round once the deadline
    passes mid-climb -- not just refuse to start in the first place."""
    p1 = DummyProblem("p1")
    problems = [p1]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=10)
    stats = {}
    policy = DummyPolicy((5,))  # max_cost starts at 4

    calls = []

    def fake_solve_step(**kwargs):
        calls.append(kwargs["complexity"])
        # NO_SOLUTION keeps max_cost unchanged (4), so the loop would otherwise keep climbing
        # (complexity 3, 4, 5, ... up to max_complexity=10) if nothing stopped it.
        return Result.NO_SOLUTION, None, []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config, **kwargs: [])

    # A fake clock: t=0 for the upfront gate, the one _test_policy_on_problems call this makes
    # before the climb starts (its own start()/per-problem/per-iteration/validationTime-end
    # checks -- see _test_policy_on_problems's wall_deadline handling), and the first round's
    # pre-check (all pass, deadline is far off), then t=1000 at the second round's pre-check
    # (deadline long passed). Real elapsed wall time in this mocked test is negligible, so a
    # hand-rolled clock is what makes the "passes mid-climb" moment deterministic instead of
    # racing the CPU.
    clock = iter([0.0] * 6)
    monkeypatch.setattr(isolver.time, "perf_counter", lambda: next(clock, 1000.0))

    best_policy, best_solved = _final_cost_minimization_pass(
        None, problems, problem_iterator, config, stats, policy, wall_deadline=100.0
    )

    # Only the complexity-3 round ran; the second (complexity 4) never started.
    assert calls == [3]
    assert best_policy is policy
    assert stats["finalPassRounds"] == 1
    assert stats["finalPassStoppedBy"] == "wall_time"


def test_final_pass_stops_between_rounds_on_a_pending_signal(monkeypatch):
    """Same as above, but for a signal arriving mid-climb instead of the deadline passing."""
    p1 = DummyProblem("p1")
    problems = [p1]
    problem_iterator = make_problem_iterator(problems, succ_complexity=2)
    config = base_config(max_complexity=10)
    stats = {}
    policy = DummyPolicy((5,))

    calls = []

    def fake_solve_step(**kwargs):
        calls.append(kwargs["complexity"])
        request_stop()  # simulate a signal arriving while this round was in flight
        return Result.NO_SOLUTION, None, []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config, **kwargs: [])

    try:
        best_policy, best_solved = _final_cost_minimization_pass(
            None, problems, problem_iterator, config, stats, policy, wall_deadline=isolver.time.perf_counter() + 3600
        )
    finally:
        reset_stop()

    assert calls == [3]
    assert best_policy is policy
    assert stats["finalPassRounds"] == 1
    assert stats["finalPassStoppedBy"] == "signal"


def test_main_loop_stops_early_by_final_pass_budget_when_the_pass_is_enabled(monkeypatch):
    """When `final_cost_minimization` is enabled (together with `add_problem_after_success`,
    its only non-no-op combination), the round loop must stop `final_pass_budget` seconds before
    `final_pass_deadline` (`max_wall_time - wall_time_reserve`), so the pass gets a guaranteed
    slice of the budget instead of only whatever the loop happens to leave behind."""
    p1 = DummyProblem("p1", objects=[1])
    problems = [p1]
    config = solve_iteratively_config(max_wall_time=1000, wall_time_reserve=100, final_pass_budget=400)

    round_deadlines = []

    def fake_solve_step(**kwargs):
        round_deadlines.append(kwargs["wall_deadline"])
        return Result.SUCCESS, DummyPolicy((2,)), []

    pass_deadlines = []

    def fake_final_pass(domain, problems, problem_iterator, config, stats, policy, wall_deadline=None):
        pass_deadlines.append(wall_deadline)
        return policy, [p1]

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config, **kwargs: [])
    monkeypatch.setattr(isolver, "_final_cost_minimization_pass", fake_final_pass)

    isolver.solve_iteratively(None, problems, config)

    assert len(round_deadlines) == 1
    assert len(pass_deadlines) == 1
    # Both deadlines are computed from the same wall_time_start at the top of the function, so
    # their difference must be final_pass_budget, independent of that unknown start time (a
    # small epsilon guards against float rounding, not against a logic error).
    assert abs((pass_deadlines[0] - round_deadlines[0]) - 400.0) < 1e-6


# --- keep_best_policy (hyp/keep-best) -------------------------------------------------------
#
# A policy learned early in a run can already generalize to nearly every problem, but
# add_problem_after_success keeps growing the training set on a handful of holdouts; later
# policies are patchworks fitted to the larger set and can solve far fewer problems overall, even
# though every success is already tested on all problems inside the loop. keep_best_policy (on by
# default) tracks the best-by-(solved, then cost) policy across the whole run -- including the
# final cost-minimization pass -- and returns that instead of whatever the run ends on.


def test_keep_best_policy_returns_the_better_round_when_the_final_pass_is_worse(monkeypatch):
    """The final pass is mocked to hand back a policy that solves strictly fewer problems than
    the round-1 policy that fed it (the same shape as the bug: a later, cheaper-looking candidate
    that covers less). With keep_best_policy on (the default), the run must still return the
    round-1 policy and its full coverage, not the final pass's output."""
    p1, p2 = DummyProblem("p1", objects=[1]), DummyProblem("p2", objects=[1, 2])
    problems = [p1, p2]
    config = solve_iteratively_config()  # keep_best_policy left at its default (true)

    def fake_solve_step(**kwargs):
        return Result.SUCCESS, DummyPolicy((2,)), []

    def fake_execute_policy(domain, problem, policy, config, **kwargs):
        return []  # round 1 solves everything -> stop_after_first_solution breaks immediately

    def fake_final_pass(domain, problems, problem_iterator, config, stats, policy, wall_deadline=None):
        # A cheaper-looking policy that only solves p1.
        return DummyPolicy((1,)), [p1]

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    monkeypatch.setattr(isolver, "_final_cost_minimization_pass", fake_final_pass)

    policy, solved, stats = isolver.solve_iteratively(None, problems, config)

    assert policy.cost == (2,)
    assert {p.name for p in solved} == {"p1", "p2"}
    assert stats["bestSolved"] == 2
    assert stats["bestCost"] == 2
    assert stats["bestRound"] == 1
    assert stats["lastSolved"] == 1  # what the final pass alone reported


def test_keep_best_policy_false_returns_the_last_policy_even_if_worse(monkeypatch):
    """keep_best_policy: false must reproduce the old behaviour exactly: the run returns whatever
    it ends on, even when an earlier round covered strictly more problems."""
    p1, p2 = DummyProblem("p1", objects=[1]), DummyProblem("p2", objects=[1, 2])
    problems = [p1, p2]
    config = solve_iteratively_config(keep_best_policy=False)

    def fake_solve_step(**kwargs):
        return Result.SUCCESS, DummyPolicy((2,)), []

    def fake_execute_policy(domain, problem, policy, config, **kwargs):
        return []

    def fake_final_pass(domain, problems, problem_iterator, config, stats, policy, wall_deadline=None):
        return DummyPolicy((1,)), [p1]

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    monkeypatch.setattr(isolver, "_final_cost_minimization_pass", fake_final_pass)

    policy, solved, stats = isolver.solve_iteratively(None, problems, config)

    assert policy.cost == (1,)
    assert [p.name for p in solved] == ["p1"]
    assert "bestSolved" not in stats
    assert "lastSolved" not in stats


def test_in_loop_test_does_not_stop_at_the_first_failing_problem_when_keeping_best(monkeypatch):
    """Unlike the pre-existing behaviour, the in-loop test must not stop at the first problem it
    cannot solve while keep_best_policy is on: the coverage count feeding the best-policy
    comparison must be exact, so every problem is tested every round."""
    p1 = DummyProblem("p1", objects=[1])
    p2 = DummyProblem("p2", objects=[1, 2])
    p3 = DummyProblem("p3", objects=[1, 2, 3])
    problems = [p1, p2, p3]
    # p1 always fails, so no round ever solves everything; the ladder eventually exhausts and
    # the loop ends via StopIteration. A generous but finite script keeps the test from hanging
    # if that took more rounds than expected -- it fails loudly (IndexError) rather than looping.
    script = [(Result.SUCCESS, DummyPolicy((2,)), []) for _ in range(50)]

    def fake_solve_step(**kwargs):
        return script.pop(0)

    exec_calls = []

    def fake_execute_policy(domain, problem, policy, config, **kwargs):
        exec_calls.append(problem.name)
        if problem.name == "p1":
            raise RuntimeError("no solution")
        return []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    config = solve_iteratively_config(final_cost_minimization=False)
    isolver.solve_iteratively(None, problems, config)

    assert "p2" in exec_calls
    assert "p3" in exec_calls


def test_in_loop_test_stops_at_the_first_failing_problem_when_not_keeping_best(monkeypatch):
    """keep_best_policy: false restores the old early-break behaviour: once p1 (the smallest,
    tested first) fails, p2 and p3 are never even tried that round -- and since every round fails
    the same way here, they are never tried at all."""
    p1 = DummyProblem("p1", objects=[1])
    p2 = DummyProblem("p2", objects=[1, 2])
    p3 = DummyProblem("p3", objects=[1, 2, 3])
    problems = [p1, p2, p3]
    script = [(Result.SUCCESS, DummyPolicy((2,)), []) for _ in range(50)]

    def fake_solve_step(**kwargs):
        return script.pop(0)

    exec_calls = []

    def fake_execute_policy(domain, problem, policy, config, **kwargs):
        exec_calls.append(problem.name)
        if problem.name == "p1":
            raise RuntimeError("no solution")
        return []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    config = solve_iteratively_config(final_cost_minimization=False, keep_best_policy=False)
    isolver.solve_iteratively(None, problems, config)

    assert "p2" not in exec_calls
    assert "p3" not in exec_calls
    assert exec_calls == ["p1"] * len(exec_calls)


def test_main_loop_uses_the_full_deadline_when_the_pass_is_disabled(monkeypatch):
    """Without the pass enabled, the round loop's own deadline must be the unnarrowed
    `final_pass_deadline` -- `final_pass_budget` only matters together with the pass."""
    p1 = DummyProblem("p1", objects=[1])
    problems = [p1]
    config = solve_iteratively_config(
        max_wall_time=1000, wall_time_reserve=100, final_pass_budget=400, final_cost_minimization=False
    )

    round_deadlines = []

    def fake_solve_step(**kwargs):
        round_deadlines.append(kwargs["wall_deadline"])
        return Result.SUCCESS, DummyPolicy((2,)), []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config, **kwargs: [])

    before = isolver.time.perf_counter()
    isolver.solve_iteratively(None, problems, config)
    after = isolver.time.perf_counter()

    assert len(round_deadlines) == 1
    # final_pass_deadline = wall_time_start + 1000 - 100 = wall_time_start + 900, and with the
    # pass disabled wall_deadline == final_pass_deadline exactly -- final_pass_budget unused.
    assert before + 900 <= round_deadlines[0] <= after + 900
