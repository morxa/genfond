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


class DummyProblem:
    def __init__(self, name, objects=()):
        self.name = name
        self.objects = objects


class DummyPolicy:
    def __init__(self, cost):
        self.cost = cost


def make_problem_iterator(active_problems, succ_complexity=2):
    return types.SimpleNamespace(
        active_problems=active_problems,
        active_plans={},
        dead_states={},
        succ_complexity=succ_complexity,
    )


def base_config(**overrides):
    config = {
        "max_complexity": 6,
        "use_unrestricted_features": False,
        "policy_iterations": 1,
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

    def fake_execute_policy(domain, problem, policy, config):
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
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config: [])

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

    def fake_execute_policy(domain, problem, policy, config):
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
    monkeypatch.setattr(isolver, "execute_policy", lambda domain, problem, policy, config: [])

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

    def fake_execute_policy(domain, problem, policy, config):
        if problem.name == "p2":
            raise RuntimeError("no action found")
        return []

    final_pass_calls = []

    def fake_final_pass(domain, problems, problem_iterator, config, stats, policy):
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
