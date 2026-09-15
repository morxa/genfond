"""Raw-ASP tests for the minimize_selected_count Occam bias (solve_datalog_sig.lp only).

SEPARATE forces two independent separation decisions, each with two ways to satisfy it: hit
distinguishing set D1 with either x1 (complexity 1) or y (complexity 3), and hit D2 with either
x2 (complexity 1) or y (complexity 3). Selecting {x1, x2} costs complexity 2 (count 2); selecting
{y} alone costs complexity 3 but count 1 -- y alone hits both D1 and D2. Both states force exactly
one candidate to be good (the alternative transitions target an unexpanded/dead state, which
`:- good_trans(...), not alive(S2), not pruned(S2)` forbids selecting), so which signature is
good/bad is not a choice -- only which concepts get selected to separate them is.

With minimize_selected_count="none" (or "below", since {x1, x2} is the unique cheapest solution
and there is no tie to break), the plain complexity #minimize picks {x1, x2} (cost 2 < 3). With
"above", the selected-element count is decided *before* complexity, and {y} alone has strictly
fewer selected elements (1 vs 2), so it wins even though it costs more raw complexity (3 vs 2) --
this is the "several cheap narrow elements vs. one general one" trade-off from
docs/min-count-results.md, reproduced in miniature.
"""

import itertools

import pytest

from genfond.config_handler import ConfigHandler
from genfond.cost_utils import feature_cost
from genfond.iterative_solver import solve
from genfond.problem_iterator import MAX_COST
from genfond.solver import Solver

SEPARATE = """
    state(0, 100).
    alive(0, 100).
    state(0, 101).
    alive(0, 101).

    state(0, 110).
    alive(0, 110).
    goal(0, 110).

    trans(0, 100, "g1(x)", 110).
    trans(0, 100, "d1(x)", 120).
    trans(0, 101, "g2(x)", 110).
    trans(0, 101, "d2(x)", 121).

    asig(0, 100, "g1(x)", "kg1").
    asig(0, 100, "d1(x)", "kb1").
    asig(0, 101, "g2(x)", "kg2").
    asig(0, 101, "d2(x)", "kb2").

    sig_pair("kg1", "kb1", "D1").
    dist("D1", c("x1")).
    dist("D1", c("y")).

    sig_pair("kg2", "kb2", "D2").
    dist("D2", c("x2")).
    dist("D2", c("y")).

    concept("x1").
    concept_complexity("x1", 1).
    concept("x2").
    concept_complexity("x2", 1).
    concept("y").
    concept_complexity("y", 3).
"""

# The good-signature tie from tests/test_minimize_good_signatures.py's TIE, reused verbatim
# (with its own, disjoint state/instance ids) to check that minimize_selected_count composes
# with minimize_good_signatures: {a, a} (1 good signature: ka) is cheaper by signature count than
# {b, c} (2), and neither needs any concept -- so it exercises a completely independent axis from
# SEPARATE's x1/x2/y trade-off within the same grounded program.
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
"""

COMBINED = TIE + SEPARATE


def sig_solver(program: str, **kwargs) -> Solver:
    return Solver(program, solve_prog="solve_datalog_sig.lp", **kwargs)


def selected_concepts(solver: Solver) -> set:
    return {c for c in solver.solution.get("c_selected", set())} - {"name"}


def test_minimize_selected_count_none_is_untouched():
    solver = sig_solver(SEPARATE)
    assert solver.solve()
    assert selected_concepts(solver) == {"x1", "x2"}
    # No #minimize over selected count is grounded with "none", so the cost vector is exactly
    # what it was before this feature existed: one entry, the feature/concept/role complexity.
    assert solver.cost == [2]
    assert feature_cost(solver.cost) == 2
    assert solver.feature_complexity == 2


def test_minimize_selected_count_below_matches_baseline_when_no_tie():
    solver = sig_solver(SEPARATE, minimize_selected_count="below")
    assert solver.solve()
    # {x1, x2} is the unique cheapest-complexity solution (2 < 3), so there is no tie for
    # "below" to break -- it reproduces the baseline choice.
    assert selected_concepts(solver) == {"x1", "x2"}
    # Complexity (2, @0) stays second to last; the new last entry is the selected-element count
    # (3 = "name" + x1 + x2, @-2).
    assert solver.cost == [2, 3]
    assert feature_cost(solver.cost, minimize_selected_count="below") == 2
    assert solver.feature_complexity == 2


def test_minimize_selected_count_above_prefers_one_expensive_element():
    solver = sig_solver(SEPARATE, minimize_selected_count="above")
    assert solver.solve()
    # {y} alone hits both distinguishing sets with 1 selected element (name + y = 2 total)
    # instead of 2 (name + x1 + x2 = 3), even though its raw complexity (3) is higher than
    # {x1, x2}'s (2) -- the point of the "above" setting.
    assert selected_concepts(solver) == {"y"}
    # The selected-element count (2, @1) leads the cost vector; complexity (3, @0) stays last,
    # unlike with "below".
    assert solver.cost == [2, 3]
    assert feature_cost(solver.cost, minimize_selected_count="above") == 3
    assert solver.feature_complexity == 3


def test_minimize_selected_count_rejects_unknown_value():
    with pytest.raises(ValueError):
        sig_solver(SEPARATE, minimize_selected_count="sideways")


def test_minimize_selected_count_requires_signature_solve_prog(gripper):
    """c_selected/f_selected/r_selected exist in solve_datalog.lp too, but the
    minimize_selected_count_{below,above} #program parts only exist in solve_datalog_sig.lp;
    applying the option to another solve_prog must fail loudly instead of an opaque clingo
    grounding error."""
    domain, problem = gripper
    config = ConfigHandler(type="datalog")
    config["minimize_selected_count"] = "above"
    with pytest.raises(ValueError):
        solve(domain, [problem], config=config, complexity=2, max_cost=MAX_COST)


@pytest.mark.parametrize(
    "minimize_good_signatures,minimize_selected_count",
    list(itertools.product(["none", "below", "above"], repeat=2)),
)
def test_minimize_selected_count_composes_with_minimize_good_signatures(
    minimize_good_signatures, minimize_selected_count
):
    """All nine combinations must ground and solve, and cost_utils.feature_cost (via
    Solver.feature_complexity) must always recover the true feature/concept/role complexity --
    2 when {x1, x2} is selected, 3 when {y} is selected -- regardless of which axis, if any, is
    biased and in which direction. This is the "level layout computed from the config, not
    hard-coded" property.
    """
    solver = sig_solver(
        COMBINED,
        minimize_good_signatures=minimize_good_signatures,
        minimize_selected_count=minimize_selected_count,
    )
    assert solver.solve()
    # The selected-element axis (SEPARATE) is independent of the good-signature axis (TIE):
    # {y} alone is chosen iff minimize_selected_count == "above", regardless of the other axis.
    expected_concepts = {"y"} if minimize_selected_count == "above" else {"x1", "x2"}
    assert selected_concepts(solver) == expected_concepts
    expected_complexity = 3 if minimize_selected_count == "above" else 2
    assert solver.feature_complexity == expected_complexity
    # The good-signature axis (TIE) is only determinate when it is actually biased; "none"
    # leaves the choice between {ka} and {kb, kc} unconstrained (either is equally good), so it
    # is not asserted in that case.
    if minimize_good_signatures != "none":
        good = {k for k, _ in solver.solution["sig_action"] if k in ("ka", "kb", "kc")}
        assert good == {"ka"}
