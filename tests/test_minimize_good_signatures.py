"""Raw-ASP tests for the minimize_good_signatures Occam bias (solve_datalog_sig.lp only).

TIE sets up a genuine tie: state 0 offers actions a/b to disjoint goal states, state 1 offers
a/c to (different) disjoint goal states, and a's signature class ("ka") is shared between the
two states while b's and c's ("kb", "kc") are each private to one. Exactly two combinations are
feasible -- {a,a} (1 good signature: ka) or {b,c} (2 good signatures: kb, kc) -- because picking
a at only one of the two states leaves the other's "a" transition unselected, which flags ka as
bad while it is also good elsewhere, violating `:- good_sig(K), bad_sig(K)`. Both combinations
need zero selected concepts/roles/features (a and b, and a and c, are different action names, so
neither pair needs a distinguishing feature at all), so with minimize_good_signatures="none"
either is an optimal model; "below" and "above" must both settle on {a,a} since it is cheaper by
signature count and tied by feature cost either way.
"""

import pytest

from genfond.config_handler import ConfigHandler
from genfond.cost_utils import feature_cost
from genfond.iterative_solver import solve
from genfond.problem_iterator import MAX_COST
from genfond.solver import Solver

TIE = """
    state(0, 0).
    alive(0, 0).
    state(0, 1).
    alive(0, 1).

    state(0, 10).
    alive(0, 10).
    goal(0, 10).
    state(0, 11).
    alive(0, 11).
    goal(0, 11).
    state(0, 12).
    alive(0, 12).
    goal(0, 12).
    state(0, 13).
    alive(0, 13).
    goal(0, 13).

    trans(0, 0, "a(x)", 10).
    trans(0, 0, "b(x)", 11).
    trans(0, 1, "a(x)", 12).
    trans(0, 1, "c(x)", 13).

    asig(0, 0, "a(x)", "ka").
    asig(0, 0, "b(x)", "kb").
    asig(0, 1, "a(x)", "ka").
    asig(0, 1, "c(x)", "kc").

    % An unused, optional concept: with nothing at all costed, the feature/concept/role
    % #minimize has no grounded element and clingo omits the whole level from the cost vector
    % (the same reason the frontier level is absent without pruned/2, see AGENTS.md). This
    % keeps that level present at cost 0, matching a real run.
    concept("c_dummy").
    concept_complexity("c_dummy", 1).
"""


def sig_solver(program: str, **kwargs) -> Solver:
    return Solver(program, solve_prog="solve_datalog_sig.lp", **kwargs)


def test_minimize_good_signatures_none_is_untouched():
    solver = sig_solver(TIE)
    assert solver.solve()
    # No #minimize over good_sig is grounded with "none", so the cost vector is exactly what it
    # was before this feature existed: one entry, the feature/concept/role complexity.
    assert solver.cost == [0]
    assert feature_cost(solver.cost) == 0
    assert solver.feature_complexity == 0


def test_minimize_good_signatures_below_prefers_fewer_signatures():
    solver = sig_solver(TIE, minimize_good_signatures="below")
    assert solver.solve()
    good = {k for k, _ in solver.solution["sig_action"]}
    assert good == {"ka"}
    # Feature complexity (0, @0) stays second to last; the new last entry is the good-signature
    # count (1, @-1) -- it is decided after feature cost, so cost[-1] is no longer the feature
    # complexity here.
    assert solver.cost == [0, 1]
    assert feature_cost(solver.cost, "below") == 0
    assert solver.feature_complexity == 0


def test_minimize_good_signatures_above_prefers_fewer_signatures():
    solver = sig_solver(TIE, minimize_good_signatures="above")
    assert solver.solve()
    good = {k for k, _ in solver.solution["sig_action"]}
    assert good == {"ka"}
    # The good-signature count (1, @1) is now decided before feature cost (0, @0), so it leads
    # the cost vector; feature complexity stays last, unlike with "below".
    assert solver.cost == [1, 0]
    assert feature_cost(solver.cost, "above") == 0
    assert solver.feature_complexity == 0


def test_minimize_good_signatures_rejects_unknown_value():
    with pytest.raises(ValueError):
        sig_solver(TIE, minimize_good_signatures="sideways")


def test_minimize_good_signatures_requires_signature_solve_prog(gripper):
    """good_sig/1 and the minimize_good_sigs_* program parts only exist in
    solve_datalog_sig.lp; applying the option to another solve_prog must fail loudly instead of
    letting clingo raise an opaque grounding error."""
    domain, problem = gripper
    config = ConfigHandler(type="datalog")
    config["minimize_good_signatures"] = "above"
    with pytest.raises(ValueError):
        solve(domain, [problem], config=config, complexity=2, max_cost=MAX_COST)
