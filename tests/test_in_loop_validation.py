"""Unit tests for the cheap in-loop validation used by `solve_iteratively`'s testing block and
`_final_cost_minimization_pass`'s candidate comparisons (hyp/cheap-validation).

These mock `execute_policy` rather than driving a real domain through clingo -- see
docs/cheap-validation-results.md for the PDDL-level validation runs on the workstation that
motivated this change: an unbounded, `policy_iterations`-times-per-problem in-loop test spent
8.5h testing candidates on a 141-problem suite once `keep_best_policy` stopped the early break at
the first failing problem.

`test_execute_datalog_policy.py::test_execute_datalog_policy_time_limit_triggers_timeout` covers
the real (non-mocked) deadline mechanism inside `execute_datalog_policy`'s own loop.
"""

from genfond import iterative_solver as isolver
from genfond.execute_rule_policy import ExecutionTimeout
from genfond.iterative_solver import _test_policy_on_problems, solve_iteratively
from genfond.problem_iterator import Result
from genfond.shutdown import request_stop, reset_stop


class DummyProblem:
    def __init__(self, name, objects=()):
        self.name = name
        self.objects = objects


class DummyPolicy:
    def __init__(self, cost=(1,)):
        self.cost = cost


class DummyProblemIterator:
    """Records `set_solved` calls without pulling in the real `ProblemIterator`."""

    def __init__(self):
        self.solved: dict[str, bool] = {}

    def set_solved(self, problem, solved=True):
        self.solved[problem.name] = solved


def base_config(**overrides):
    config = {
        "validation_iterations": 1,
        "validation_max_consecutive_failures": None,
        "validation_time_limit": None,
    }
    config.update(overrides)
    return config


# --- (a) one iteration per problem ----------------------------------------------------------


def test_validation_iterations_one_calls_execute_policy_once_per_problem(monkeypatch):
    """`validation_iterations: 1` (the default) must call `execute_policy` exactly once per
    problem, not `policy_iterations` times -- the whole point of this config key is to make the
    in-loop test cheap regardless of how high `policy_iterations` (the final-verification knob)
    is set."""
    problems = [DummyProblem("p1"), DummyProblem("p2"), DummyProblem("p3")]
    calls: list[str] = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        calls.append(problem.name)
        return []

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    # A high policy_iterations must have no effect on the in-loop test.
    config = base_config(policy_iterations=100)

    solved = _test_policy_on_problems(None, problems, DummyPolicy(), config)

    assert calls == ["p1", "p2", "p3"]
    assert [p.name for p in solved] == ["p1", "p2", "p3"]


def test_validation_iterations_above_one_repeats_each_problem(monkeypatch):
    """A higher validation_iterations (meaningful for FOND, where execution samples a random
    outcome each run) must repeat every problem that many times."""
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    calls: list[str] = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        calls.append(problem.name)
        return []

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    config = base_config(validation_iterations=4)

    _test_policy_on_problems(None, problems, DummyPolicy(), config)

    assert calls == ["p1"] * 4 + ["p2"] * 4


# --- (b) stop after N consecutive failures ---------------------------------------------------


def test_stops_after_max_consecutive_failures_and_reports_a_lower_bound(monkeypatch):
    """p1 solves, p2 and p3 fail (two in a row), p4 and p5 are never reached: with
    max_consecutive_failures=2 the test stops right after p3, so the returned solved list is a
    lower bound -- p4/p5 are never executed at all, even though nothing here says they'd fail
    too."""
    p1, p2, p3, p4, p5 = (DummyProblem(n) for n in ("p1", "p2", "p3", "p4", "p5"))
    problems = [p1, p2, p3, p4, p5]
    calls: list[str] = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        calls.append(problem.name)
        if problem.name in ("p2", "p3"):
            raise RuntimeError("no solution")
        return []

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    config = base_config(validation_max_consecutive_failures=2)
    stats: dict = {}
    problem_iterator = DummyProblemIterator()

    solved = _test_policy_on_problems(
        None,
        problems,
        DummyPolicy(),
        config,
        max_consecutive_failures=2,
        problem_iterator=problem_iterator,
        stats=stats,
    )

    # p4 and p5 are never even attempted -- the test stopped as soon as p2, p3 failed in a row.
    assert calls == ["p1", "p2", "p3"]
    assert [p.name for p in solved] == ["p1"]
    assert stats["validationEarlyStops"] == 1
    assert "validationTime" in stats
    # Only the problems actually tested are reported back to the problem iterator.
    assert problem_iterator.solved == {"p1": True, "p2": False, "p3": False}


def test_a_single_failure_between_successes_does_not_trigger_the_stop(monkeypatch):
    """The gate counts *consecutive* failures: an isolated miss surrounded by successes must not
    cut the test short."""
    p1, p2, p3 = (DummyProblem(n) for n in ("p1", "p2", "p3"))
    problems = [p1, p2, p3]
    calls: list[str] = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        calls.append(problem.name)
        if problem.name == "p2":
            raise RuntimeError("no solution")
        return []

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    config = base_config()
    stats: dict = {}

    solved = _test_policy_on_problems(None, problems, DummyPolicy(), config, max_consecutive_failures=2, stats=stats)

    assert calls == ["p1", "p2", "p3"]
    assert [p.name for p in solved] == ["p1", "p3"]
    assert "validationEarlyStops" not in stats


def test_max_consecutive_failures_none_tests_every_problem(monkeypatch):
    """None (the default `_test_policy_on_problems` uses, and what
    `_final_cost_minimization_pass` always passes) must test every problem regardless of how
    many failures in a row occur -- the exact-count behaviour candidate comparison needs."""
    problems = [DummyProblem(n) for n in ("p1", "p2", "p3", "p4")]
    calls: list[str] = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        calls.append(problem.name)
        raise RuntimeError("no solution")

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    config = base_config()

    solved = _test_policy_on_problems(None, problems, DummyPolicy(), config)

    assert calls == ["p1", "p2", "p3", "p4"]
    assert solved == []


# --- (c) the time limit marks a slow problem as failed ---------------------------------------


def test_execution_timeout_counts_as_a_failure(monkeypatch):
    """ExecutionTimeout is a RuntimeError subclass raised by execute_rule_policy/
    execute_datalog_policy when a single execution runs past validation_time_limit (see
    execute_rule_policy.ExecutionTimeout and test_execute_datalog_policy_time_limit_triggers_
    timeout for the real, non-mocked mechanism); here it is simulated directly to prove
    `_test_policy_on_problems` treats it exactly like any other execution failure -- and,
    crucially, that `time_limit` is threaded through to `execute_policy` at all."""
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    seen_time_limits: list = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        seen_time_limits.append(time_limit)
        if problem.name == "p1":
            raise ExecutionTimeout({}, frozenset(), time_limit)
        return []

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    config = base_config(validation_time_limit=5)

    solved = _test_policy_on_problems(None, problems, DummyPolicy(), config)

    assert seen_time_limits == [5, 5]
    assert [p.name for p in solved] == ["p2"]


# --- graceful stop: wall_deadline / stop_requested() checked inside a single validation call --
#
# Four 12h SLURM jobs (barman, grid, reward, spanner) were killed without ever writing a stats
# row or policy file: a single call to `_test_policy_on_problems` -- one round's whole in-loop
# test -- ran for hours on its own, well past `--max-wall-time`, because nothing checked the
# deadline or a pending stop request until the *next* round's top-of-loop check, which never
# arrived. These tests cover the fix: the check now runs between problems and between repeated
# iterations of the same problem, inside the call itself.


def test_stops_between_problems_when_wall_deadline_passes(monkeypatch):
    problems = [DummyProblem("p1"), DummyProblem("p2"), DummyProblem("p3")]
    calls: list[str] = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        calls.append(problem.name)
        return []

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    # A hand-rolled clock, tied to the exact time.perf_counter() call order inside
    # _test_policy_on_problems: start() (t=0), p1's pre-problem check (t=0), p1's
    # pre-iteration check (t=0), p2's pre-problem check (t=200, past the deadline of 100) --
    # p2 is never executed. Any later call (the final validationTime bookkeeping) falls back to
    # the iterator's default of 200.
    clock = iter([0.0, 0.0, 0.0, 200.0])
    monkeypatch.setattr(isolver.time, "perf_counter", lambda: next(clock, 200.0))
    config = base_config()
    stats: dict = {}

    solved = _test_policy_on_problems(None, problems, DummyPolicy(), config, stats=stats, wall_deadline=100.0)

    assert calls == ["p1"]
    assert [p.name for p in solved] == ["p1"]
    assert stats["validationStoppedBy"] == "wall_time"


def test_stops_mid_problem_between_iterations_on_stop_requested(monkeypatch):
    """The check runs before every repeated iteration of the *same* problem too, not just
    between problems -- validation_iterations > 1 (meaningful for FOND) means a single problem's
    own repeated-execution loop could otherwise run long on its own."""
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    calls: list[str] = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        calls.append(problem.name)
        if len(calls) == 2:
            request_stop()  # simulate a signal arriving mid-test, after p1's first iteration
        return []

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)
    config = base_config(validation_iterations=3)
    stats: dict = {}
    try:
        solved = _test_policy_on_problems(None, problems, DummyPolicy(), config, stats=stats)
    finally:
        reset_stop()

    # p1's 3rd iteration and all of p2 are skipped once the stop request lands.
    assert calls == ["p1", "p1"]
    assert solved == []  # p1's own outcome is unknown -- not recorded either way
    assert stats["validationStoppedBy"] == "signal"


def test_solve_iteratively_ends_gracefully_when_validation_receives_a_stop_request(monkeypatch):
    """End to end: a stop request (the same mechanism SIGINT/SIGTERM use, see
    genfond.shutdown) landing *mid-validation* -- after p1's test but before p2's -- must end
    the run gracefully (stats["stoppedBy"] set, whatever policy already exists intact) instead
    of falling through to the ladder's ordinary "not every problem solved this round, escalate
    and try again" branch and starting another (possibly very expensive) round. Without the
    fix, nothing inside the validation call checked this, so the run would only have noticed at
    the *next* round's top-of-loop check -- which never arrives if that next round is the one
    that runs for hours (see this module's module docstring)."""
    p1, p2 = DummyProblem("p1"), DummyProblem("p2")
    problems = [p1, p2]
    config = {
        "min_complexity": 2,
        "max_complexity": 6,
        "use_unrestricted_features": False,
        "reset_complexity_on_state_space_change": False,
        "add_problem_after_success": False,
        "unselect_problems": False,
        "min_number_of_plans": 5,
        "max_frontier_expansions": 20,
        "use_example_plans": False,
        "frontier_expansion": False,
        "policy_iterations": 1,
        "validation_iterations": 1,
        "validation_max_consecutive_failures": None,
        "validation_time_limit": None,
        "stop_after_first_solution": True,
        "final_cost_minimization": False,
        "minimize_good_signatures": "none",
        "minimize_selected_count": "none",
        "keep_best_policy": True,
    }

    solve_step_calls = []

    def fake_solve_step(**kwargs):
        solve_step_calls.append(kwargs["complexity"])
        return Result.SUCCESS, DummyPolicy((2,)), []

    tested: list[str] = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        tested.append(problem.name)
        if problem.name == "p1":
            request_stop()  # a signal lands right after p1's test, before p2's
        return []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    try:
        policy, solved, stats = solve_iteratively(None, problems, config)
    finally:
        reset_stop()

    assert tested == ["p1"]  # p2's test never ran once the stop request landed
    assert len(solve_step_calls) == 1  # no second round was ever attempted
    assert policy is not None and policy.cost == (2,)  # round 1's policy is not lost
    assert stats["stoppedBy"] == "signal"
    assert "validationStoppedBy" not in stats  # renamed to stoppedBy, not left duplicated
