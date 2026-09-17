"""Helpers for reading a clingo `model.cost` vector.

Split out of `solver.py` so that `problem_iterator.py` can use it too without a circular import
(`solver.py` already imports `MAX_COST` from `problem_iterator.py`).
"""

from typing import Sequence


def _below_levels(minimize_good_signatures: str, minimize_selected_count: str) -> int:
    """How many `#minimize` levels sit *below* feature/concept/role complexity in cost().

    `model.cost` is ordered from the highest-priority `#minimize` level to the lowest, and a
    level that never grounds is absent from the vector entirely rather than reporting 0 -- see
    the cost-vector gotcha in AGENTS.md. The feature/concept/role complexity minimize in
    solve_datalog_sig.lp (and every other solve*.lp) sits at the default priority 0.

    Two independent Occam biases in solve_datalog_sig.lp can each be pinned either *above*
    complexity (a higher clingo priority, decided first) or *below* it (a lower priority, a
    pure tie-break decided after): `minimize_good_signatures` (the good-signature/rule count)
    and `minimize_selected_count` (the selected-element count). An "above" setting is grounded
    at a priority between complexity (@0) and the frontier-transition count (the highest
    priority, see solve_datalog_sig.lp), so it never moves complexity out of last place. A
    "below" setting is grounded at a priority under @0, so it *does* -- and each active "below"
    setting pushes complexity one further position from the end of the vector. Their relative
    order among themselves does not matter here, only the count does.

    Both "below" groups are guaranteed to ground whenever their `#program` part is grounded at
    all: `good_sig/1` grounds for any round with at least one alive non-goal state (true of
    every real solve), and the selected-element count grounds unconditionally because
    `c_selected(name)` is a plain fact in solve_datalog_sig.lp, not a choice. So the offset from
    the end of the cost vector is determined by the config alone, not by what the model actually
    selected.
    """
    return (minimize_good_signatures == "below") + (minimize_selected_count == "below")


def feature_cost(
    cost: Sequence[int],
    minimize_good_signatures: str = "none",
    minimize_selected_count: str = "none",
) -> int:
    """Pick the feature/concept/role complexity out of a clingo cost vector.

    See `_below_levels` for why the position depends on `minimize_good_signatures` and
    `minimize_selected_count`. With both "none" or "above" (the only settings that existed
    before `minimize_selected_count`), nothing sits below complexity and it stays `cost[-1]`.
    """
    if not cost:
        return 0
    below = _below_levels(minimize_good_signatures, minimize_selected_count)
    if below and len(cost) > below:
        return cost[-(below + 1)]
    return cost[-1]


def _above_levels(minimize_good_signatures: str, minimize_selected_count: str) -> int:
    """How many `#minimize` levels sit *between* the frontier count and complexity in cost().

    The mirror image of `_below_levels`: an "above" setting grounds its bias at a priority
    between complexity (@0) and the frontier-transition count (the highest priority), so it adds
    one element to the vector ahead of complexity but behind the frontier count.
    """
    return (minimize_good_signatures == "above") + (minimize_selected_count == "above")


def prune_cost(
    cost: Sequence[int],
    minimize_good_signatures: str = "none",
    minimize_selected_count: str = "none",
) -> int:
    """The frontier-transition count out of a clingo cost vector; 0 when that level is absent.

    `#minimize { 1@3,I,S1,A,S2 : good_trans(I,S1,A,S2), pruned(I,S2) }` (solve_datalog_sig.lp;
    `1@2` in solve_datalog.lp -- the number differs, its being the *highest* priority does not)
    only grounds when the instance has a `pruned/2` state reachable by a transition. When it does
    not, the level is missing from `model.cost` entirely rather than reporting 0, exactly as the
    cost-vector gotcha in AGENTS.md describes, and `cost[0]` is then the feature complexity (or
    one of the "above" biases) instead. So the level cannot be read off position 0 blindly.

    The other levels are all accounted for by the config: `1 + _below_levels(...)` sit at or
    below complexity and `_above_levels(...)` between, and each of them is guaranteed to ground
    whenever its `#program` part is (see `_below_levels`). Hence the vector is exactly one
    element longer than that sum iff the frontier level grounded, and that element is `cost[0]`.
    """
    if not cost:
        return 0
    without_frontier = _above_levels(minimize_good_signatures, minimize_selected_count) + 1
    without_frontier += _below_levels(minimize_good_signatures, minimize_selected_count)
    return cost[0] if len(cost) > without_frontier else 0
