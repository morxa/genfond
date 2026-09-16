"""Enumerating several optimal models per round and keeping the one with the best coverage.

clingo returns whichever optimal model it happened to prove first, and equal-cost models are not
equally good policies: on blocks3ops a cost-3 model on `c_equal_closure(on, on_g)` solves 95/95
while the equally cheap `c_equal(on, on_g)` solves 28/95, and nothing in the encoding tells them
apart (docs/opt-enum-results.md). `optimal_model_limit` enumerates up to that many optimal models
of the proven optimal cost, validates each and keeps the best.

The three claims tested here:

(a) `Solver.enumerate_optimal` returns several distinct models of the *same* cost, and exactly
    one when the limit is 1 (the default, which must not solve a second time at all).
(b) `_choose_candidate` picks by coverage, then rule count, then clingo's order -- and
    `solve_iteratively` reuses the winner's validation instead of paying for it twice.
(c) the lazy-pairs loop discards an enumerated model that violates a pair: the enumeration runs
    on a relaxation, so a sibling of the incumbent need not be feasible for the full problem.
"""

from typing import Any

import pytest

from genfond import iterative_solver as isolver
from genfond.iterative_solver import _choose_candidate, solve_iteratively
from genfond.lazy_pairs import solve_with_lazy_pairs
from genfond.problem_iterator import Result
from genfond.solver import Solver, SolveStatus, solution_key
from tests.test_lazy_pairs import INSTANCE, SIGNATURES, SOLVE_PROG

# Two features of complexity 1, `g` and `h`, distinguish the good transition `b` from the bad
# transition `a` in exactly the same way; `f` changes on both, so it distinguishes nothing. The
# optimum is therefore cost 1 with two models -- select `g`, or select `h`.
TWO_OPTIMA = """
    feature(f). feature_complexity(f, 1).
    feature(g). feature_complexity(g, 1).
    feature(h). feature_complexity(h, 1).

    state(0, 0). alive(0, 0).
    eval(0, 0, f, 0). eval(0, 0, g, 0). eval(0, 0, h, 0).

    state(0, 1).
    eval(0, 1, f, 1). eval(0, 1, g, 0). eval(0, 1, h, 0).

    state(0, 2). alive(0, 2). goal(0, 2).
    eval(0, 2, f, 1). eval(0, 2, g, 1). eval(0, 2, h, 1).

    trans(0, 0, a, 1).
    trans(0, 0, b, 2).
"""


# --- (a) the enumeration itself --------------------------------------------------------------


def test_limit_one_returns_only_the_model_the_plain_solve_found():
    """The default must be a pure no-op: the incumbent, and no second clingo solve."""
    solver = Solver(TWO_OPTIMA, num_threads=1)
    assert solver.solve()
    incumbent = solver.solution
    calls: list[Any] = []
    original = solver.control.solve
    solver.control.solve = lambda *args, **kwargs: calls.append(kwargs) or original(*args, **kwargs)  # type: ignore[method-assign]

    assert solver.enumerate_optimal(1) == [incumbent]
    assert calls == []


def test_limit_two_returns_two_distinct_models_of_the_same_cost():
    solver = Solver(TWO_OPTIMA, num_threads=1)
    assert solver.solve()
    assert solver.status == SolveStatus.OPTIMAL
    optimal_cost = solver.cost

    candidates = solver.enumerate_optimal(2)

    assert len(candidates) == 2
    assert candidates[0] is solver.solution, "the incumbent must stay first"
    assert len({solution_key(candidate) for candidate in candidates}) == 2, "models must be distinct"
    assert [candidate["cost"] for candidate in candidates] == [optimal_cost, optimal_cost]
    assert {frozenset(candidate["selected"]) for candidate in candidates} == {frozenset({"g"}), frozenset({"h"})}
    assert solver.num_enumerated == 2


def test_enumeration_leaves_the_control_usable_for_another_solve():
    """The lazy-pairs loop solves the same Control over and over, so `opt_mode`/`models` must be
    exactly as they were -- an `optN`/bounded-model configuration left behind would silently
    change every later solve."""
    solver = Solver(TWO_OPTIMA, num_threads=1)
    assert solver.solve()
    solver.enumerate_optimal(2)

    assert solver.solve()
    assert solver.status == SolveStatus.OPTIMAL
    assert solver.cost == [1]


def test_enumeration_is_skipped_when_the_solve_proved_nothing():
    """An unsatisfiable program has no incumbent and nothing to enumerate; the enumeration must
    not turn that into a candidate list."""
    solver = Solver("state(0, 0). alive(0, 0).", num_threads=1)
    assert solver.solve() is False
    assert solver.enumerate_optimal(3) == []


# --- (b) choosing between the candidates ------------------------------------------------------


class FakePolicy:
    def __init__(self, name, rules):
        self.name = name
        self.rules = frozenset(range(rules))
        self.cost = (3,)

    def __repr__(self):
        return f"FakePolicy({self.name})"


def test_choose_candidate_picks_the_highest_coverage():
    """The whole point: the equal-cost models are indistinguishable to the solver, so the one
    that solves the most problems wins even though it is neither first nor smallest."""
    first, second, third = FakePolicy("first", 2), FakePolicy("second", 9), FakePolicy("third", 3)
    coverage = {"first": ["p1"], "second": ["p1", "p2", "p3"], "third": ["p1", "p2"]}
    stats: dict = dict()

    chosen = _choose_candidate([first, second, third], lambda policy: coverage[policy.name], stats)

    assert chosen is second
    assert stats["optimalModelChosenCoverage"] == 3
    assert stats["optimalModelChosenIndex"] == 1
    assert stats["optimalModelCoverages"] == [1, 3, 2]


def test_choose_candidate_breaks_coverage_ties_by_rule_count():
    big, small = FakePolicy("big", 12), FakePolicy("small", 4)
    stats: dict = dict()

    chosen = _choose_candidate([big, small], lambda policy: ["p1", "p2"], stats)

    assert chosen is small


def test_choose_candidate_falls_back_to_clingos_own_order():
    """With coverage and rule count tied, the single-model path's choice must be reproduced
    exactly -- i.e. the first model clingo returned."""
    first, second = FakePolicy("first", 5), FakePolicy("second", 5)
    stats: dict = dict()

    assert _choose_candidate([first, second], lambda policy: ["p1"], stats) is first


def test_choose_candidate_validates_every_candidate_once():
    validated: list[str] = []
    policies = [FakePolicy("a", 1), FakePolicy("b", 1), FakePolicy("c", 1)]

    def validate(policy):
        validated.append(policy.name)
        return []

    _choose_candidate(policies, validate, dict())

    assert validated == ["a", "b", "c"]


# --- (b, integration) solve_iteratively hands solve_step a validator and reuses its result ----


class DummyProblem:
    def __init__(self, name, objects=()):
        self.name = name
        self.objects = objects


def _loop_config(**overrides):
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
        "optimal_model_limit": 3,
    }
    config.update(overrides)
    return config


def test_solve_iteratively_reuses_the_winning_candidates_validation(monkeypatch):
    """`solve_step` already validated the policy it returned, so the round's own testing block
    must not execute it a second time -- but the iterator still has to learn which problems that
    policy solved."""
    p1, p2 = DummyProblem("p1"), DummyProblem("p2")
    problems = [p1, p2]
    winner = FakePolicy("winner", 1)
    loser = FakePolicy("loser", 1)

    def fake_solve_step(**kwargs):
        validate = kwargs["validate"]
        assert validate is not None, "solve_iteratively must hand solve_step a validator"
        # Both candidates are scored, exactly as the real solve_step does, and the better one
        # is returned.
        assert len(validate(loser)) == 1
        assert len(validate(winner)) == 2
        return Result.SUCCESS, winner, []

    executed: list[tuple[str, str]] = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        executed.append((policy.name, problem.name))
        if policy is loser and problem.name == "p2":
            raise RuntimeError("no action found")
        return []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    policy, solved, stats = solve_iteratively(None, problems, _loop_config())

    assert policy is winner
    assert {p.name for p in solved} == {"p1", "p2"}
    # Two candidates x two problems, and nothing more: the winner is not re-executed by the
    # round's own testing block.
    assert executed == [("loser", "p1"), ("loser", "p2"), ("winner", "p1"), ("winner", "p2")]


def test_solve_iteratively_still_tests_a_policy_solve_step_did_not_validate(monkeypatch):
    """With the default limit `solve_step` never calls the validator, so the round's testing
    block must run exactly as it always did."""
    p1 = DummyProblem("p1")

    def fake_solve_step(**kwargs):
        return Result.SUCCESS, FakePolicy("only", 1), []

    executed: list[str] = []

    def fake_execute_policy(domain, problem, policy, config, time_limit=None):
        executed.append(problem.name)
        return []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    policy, solved, stats = solve_iteratively(None, [p1], _loop_config(optimal_model_limit=1))

    assert executed == ["p1"]
    assert [p.name for p in solved] == ["p1"]


# --- (c) the lazy loop must discard infeasible siblings ---------------------------------------


def _bogus_model(cost):
    """An equal-cost "model" whose selection separates nothing: `b_f1` alone leaves the pairs
    (0, 2) and (0, 3) violated (see test_lazy_pairs.test_violated_pairs_agree_with_the_eager_
    relation). A model like this is exactly what the relaxation can produce -- the pairs it
    violates were never grounded -- and it must never become a candidate policy."""
    return {
        "f_selected": {'"b_f1"'},
        "c_selected": set(),
        "r_selected": set(),
        "sig_action": {(0, '"a(x0)"')},
        "bad_sig": {1, 2, 3},
        "cost": cost,
    }


def test_lazy_loop_discards_an_enumerated_model_that_violates_a_pair(monkeypatch):
    lazy = Solver(INSTANCE, num_threads=1, solve_prog=SOLVE_PROG)

    def fake_enumerate(self, limit):
        self.candidates = [self.solution, _bogus_model(self.cost)]
        self.num_enumerated = 2
        return self.candidates

    monkeypatch.setattr(Solver, "enumerate_optimal", fake_enumerate)

    status = solve_with_lazy_pairs(lazy, SIGNATURES, batch_size=1, optimal_model_limit=2)

    assert status == SolveStatus.OPTIMAL
    # The incumbent survives (the loop stops precisely because it violates nothing); the
    # relaxation-only sibling does not.
    assert lazy.candidates == [lazy.solution]
    assert lazy.num_enumerated == 2


def test_lazy_loop_keeps_a_feasible_enumerated_model(monkeypatch):
    """The filter must not be a blanket rejection of everything the enumeration returns."""
    lazy = Solver(INSTANCE, num_threads=1, solve_prog=SOLVE_PROG)

    def fake_enumerate(self, limit):
        # The incumbent's own selection under a different key order -- feasible by construction.
        twin = dict(self.solution)
        twin["marker"] = {1}
        self.candidates = [self.solution, twin]
        self.num_enumerated = 2
        return self.candidates

    monkeypatch.setattr(Solver, "enumerate_optimal", fake_enumerate)

    assert solve_with_lazy_pairs(lazy, SIGNATURES, batch_size=1, optimal_model_limit=2) == SolveStatus.OPTIMAL
    assert len(lazy.candidates) == 2


def test_lazy_loop_does_not_enumerate_at_all_with_the_default_limit(monkeypatch):
    lazy = Solver(INSTANCE, num_threads=1, solve_prog=SOLVE_PROG)

    def fail_enumerate(self, limit):
        raise AssertionError("no enumeration may happen with optimal_model_limit=1")

    monkeypatch.setattr(Solver, "enumerate_optimal", fail_enumerate)

    assert solve_with_lazy_pairs(lazy, SIGNATURES, batch_size=1) == SolveStatus.OPTIMAL
    assert lazy.candidates == []
