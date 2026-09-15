"""Helpers for reading a clingo `model.cost` vector.

Split out of `solver.py` so that `problem_iterator.py` can use it too without a circular import
(`solver.py` already imports `MAX_COST` from `problem_iterator.py`).
"""

from typing import Sequence


def feature_cost(cost: Sequence[int], minimize_good_signatures: str = "none") -> int:
    """Pick the feature/concept/role complexity out of a clingo cost vector.

    `model.cost` is ordered from the highest-priority `#minimize` level to the lowest, and a
    level that never grounds (no literal in it could ever hold) is absent from the vector
    entirely rather than reporting 0 -- see the cost-vector gotcha in AGENTS.md. The
    feature/concept/role complexity minimize in solve_datalog_sig.lp (and every other solve*.lp)
    sits at the default priority 0.

    With `minimize_good_signatures == "below"` (see solve_datalog_sig.lp), the good-signature
    count is minimized at a *lower* priority than that (@-1), specifically so it is decided
    after feature cost -- which makes it the new lowest-priority, hence last, entry of the cost
    vector, and pushes feature complexity to the second-to-last position. With "none" or
    "above" nothing sits below priority 0, so feature complexity stays last. This assumes
    `good_sig/1` grounds whenever the feature-complexity level does, which holds for any round
    with at least one alive non-goal state -- true of every real solve.
    """
    if not cost:
        return 0
    if minimize_good_signatures == "below" and len(cost) >= 2:
        return cost[-2]
    return cost[-1]
