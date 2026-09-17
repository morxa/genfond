"""The lazy separation loop must agree with the eager encoding it relaxes.

`solve_with_lazy_pairs` grounds no `sig_pair`/`dist` facts up front and adds a batch of violated
pairs per iteration instead. The claim is that the model it ends on is an optimum of the *full*
problem, so these tests pin it against the eager encoding: same cost, and the selection it
returns is feasible for the eagerly grounded program at that cost.
"""

import pytest

from genfond.action_signatures import ActionSignature, LazyPairs, iter_dist_set_facts
from genfond.config_handler import ConfigHandler
from genfond.feature_generator import FeaturePool
from genfond.lazy_pairs import solve_with_lazy_pairs
from genfond.solver import Solver, SolveStatus

MAX_COMPLEXITY = 4
SOLVE_PROG = "solve_datalog_sig.lp"


def _signature(name, arity, concepts, roles, bools):
    return ActionSignature(name, arity, frozenset(concepts), frozenset(roles), tuple(bools))


# One alive state with four actions of the same name: the first is the only one that reaches the
# goal, so it is the good class and the other three are bad. Every bad class has to be separated
# from it, and each is cheapest to separate by a *different* element, so no single selection
# closes the round and a batch size of one forces several iterations.
SIGNATURES = [
    _signature("a", 1, [], [], [("b_f1", 1), ("b_f2", 1), ("b_f3", 1)]),
    _signature("a", 1, [], [], [("b_f1", 0), ("b_f2", 1), ("b_f3", 1)]),
    _signature("a", 1, [], [], [("b_f1", 1), ("b_f2", 0), ("b_f3", 1)]),
    _signature("a", 1, [("c_c1", 0)], [], [("b_f1", 1), ("b_f2", 1), ("b_f3", 1)]),
]

INSTANCE = """
feature("b_f1"). feature_complexity("b_f1", 1).
feature("b_f2"). feature_complexity("b_f2", 2).
feature("b_f3"). feature_complexity("b_f3", 9).
concept("c_c1"). concept_complexity("c_c1", 4).
state(0, 0). state(0, 1). alive(0, 0). alive(0, 1). goal(0, 1).
trans(0, 0, "a(x0)", 1).
trans(0, 0, "a(x1)", 0). trans(0, 0, "a(x2)", 0). trans(0, 0, "a(x3)", 0).
asig(0, 0, "a(x0)", 0). asig(0, 0, "a(x1)", 1).
asig(0, 0, "a(x2)", 2). asig(0, 0, "a(x3)", 3).
"""


def _solver(instance):
    return Solver(instance, num_threads=1, solve_prog=SOLVE_PROG)


def test_lazy_loop_needs_several_iterations_and_matches_the_eager_cost():
    eager = _solver(INSTANCE + "".join(iter_dist_set_facts(SIGNATURES)))
    assert eager.solve()
    # b_f1 (1) + b_f2 (2) + c_c1 (4); b_f3 is constant, so it separates nothing.
    assert eager.cost[-1] == 7

    lazy = _solver(INSTANCE)
    stats: dict = dict()
    assert solve_with_lazy_pairs(lazy, SIGNATURES, batch_size=1, stats=stats)
    assert lazy.cost == eager.cost
    assert stats["lazyPairIterations"] >= 2
    # Only the three pairs that involve the good class are ever needed; the eager encoding
    # grounds all six.
    assert stats["lazyPairsGrounded"] < stats["lazyPairsTotal"] == 6


def test_lazy_loop_reports_an_unsatisfiable_relaxation_as_unsatisfiable():
    # No transition reaches a goal, so the graph layer alone is already unsatisfiable and no
    # separation pair can repair it.
    unsat = _solver("state(0, 0). alive(0, 0).")
    assert solve_with_lazy_pairs(unsat, SIGNATURES, batch_size=1) == SolveStatus.UNSATISFIABLE


def test_violated_pairs_agree_with_the_eager_relation():
    """A pair is violated iff it is a good/bad pair no selected element distinguishes."""
    pairs = LazyPairs(SIGNATURES)
    for selection, expected in [
        ((set(), set(), set()), {(0, 1), (0, 2), (0, 3)}),
        (({"b_f1"}, set(), set()), {(0, 2), (0, 3)}),
        (({"b_f1", "b_f2"}, set(), set()), {(0, 3)}),
        (({"b_f3"}, {"c_c1"}, set()), {(0, 1), (0, 2)}),
        (({"b_f1", "b_f2"}, {"c_c1"}, set()), set()),
    ]:
        total, violated = pairs.violated_pairs(*selection, good={0}, bad={1, 2, 3})
        assert total == len(expected)
        assert set(violated) == expected


def test_violated_pairs_are_sorted_by_the_size_of_the_distinguishing_set():
    pairs = LazyPairs(SIGNATURES)
    # Class 3 differs from the good one in a concept only, classes 1 and 2 in one feature each,
    # and class 1 additionally in nothing else -- all three sets have size one here, so make one
    # of them bigger by selecting nothing and asking for the mirrored labelling.
    _, violated = pairs.violated_pairs(set(), set(), set(), good={1}, bad={0, 2})
    # {1, 2} differ in b_f1 and b_f2, {0, 1} only in b_f1.
    assert violated == [(1, 0), (1, 2)]


@pytest.mark.parametrize("fixture", ["gripper", "blocks_clear"])
def test_the_lazy_selection_is_feasible_for_the_eagerly_grounded_program(fixture, request):
    """The end-to-end correctness claim, checked without trusting the Python pair scan.

    The lazy loop stops when *its own* computation finds no violated pair, so pinning it against
    that computation would be circular. Instead the selection it returns is forced onto the
    eagerly grounded program, which carries every pair as an ASP constraint: it must still be
    satisfiable, and at the same cost.
    """
    domain, problem = request.getfixturevalue(fixture)
    config = ConfigHandler(type="datalog-sig")
    config["lazy_pairs"] = False
    pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
    eager_instance = pool.to_clingo()
    eager = _solver(eager_instance)
    assert eager.solve()

    config["lazy_pairs"] = True
    lazy_pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
    lazy = _solver(lazy_pool.to_clingo())
    assert solve_with_lazy_pairs(lazy, lazy_pool.signatures, batch_size=config["lazy_pairs_batch"])
    assert lazy.cost[-1] == eager.cost[-1]

    forced = "".join(
        f'{key}("{element}").'
        # `name` is the identity concept the program selects unconditionally, as a constant
        # rather than a string; it distinguishes nothing.
        for key in ("f_selected", "c_selected", "r_selected")
        for element in (str(symbol).strip('"') for symbol in lazy.solution.get(key, set()))
        if element != "name"
    )
    checked = _solver(eager_instance + forced)
    assert checked.solve(), "the lazy selection violates a pair the eager encoding grounds"
    assert checked.cost[-1] == lazy.cost[-1]


class _CutOffSolver:
    """A `Solver` that reports every solve as cut off by its time budget.

    `solve_time_limit` can cancel a solve while it holds a model that satisfies every constraint
    of the current relaxation but whose cost was never proved minimal. lazy_pairs only reads
    `status`, `solution`, `cost` and `optimal` off the solver, so downgrading the status of a
    real solve reproduces that case exactly, without depending on clingo losing a race.
    """

    def __init__(self, solver):
        self._solver = solver

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def solve(self, bound=None):
        found = self._solver.solve(bound=bound)
        if found:
            self._solver.status = SolveStatus.SATISFIABLE
            self._solver.optimal = False
        return found


def test_a_cut_off_solve_still_yields_a_feasible_selection_but_is_not_optimal(gripper):
    """The soundness claim for the anytime path, checked against the eager encoding.

    The loop stops only when a model violates no pair, so its result is feasible for the full
    problem however many solves were cut off on the way; what it loses is the optimality
    argument, and it must say so.
    """
    domain, problem = gripper
    config = ConfigHandler(type="datalog-sig")
    config["lazy_pairs"] = False
    pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
    eager_instance = pool.to_clingo()

    config["lazy_pairs"] = True
    lazy_pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
    lazy = _CutOffSolver(_solver(lazy_pool.to_clingo()))
    status = solve_with_lazy_pairs(lazy, lazy_pool.signatures, batch_size=config["lazy_pairs_batch"])
    assert status == SolveStatus.SATISFIABLE

    forced = "".join(
        f'{key}("{element}").'
        for key in ("f_selected", "c_selected", "r_selected")
        for element in (str(symbol).strip('"') for symbol in lazy.solution.get(key, set()))
        if element != "name"
    )
    checked = _solver(eager_instance + forced)
    assert checked.solve(), "a selection the cut-off loop accepted violates an eagerly grounded pair"


class _NoModelSolver:
    """A `Solver` whose solve is cancelled before it ever reports a model."""

    def __init__(self, solver):
        self._solver = solver

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def solve(self, bound=None):
        self._solver.solution = dict()
        self._solver.cost = []
        self._solver.status = SolveStatus.UNKNOWN
        self._solver.optimal = False
        self._solver.timed_out = True
        return False


def test_a_solve_cut_off_before_its_first_model_is_reported_as_unknown():
    # The relaxation here is plainly satisfiable, so UNKNOWN must not be confused with
    # UNSATISFIABLE: the round refutes nothing and the caller must escalate, not record a
    # refutation of the complexity level.
    plain = _solver(INSTANCE)
    assert solve_with_lazy_pairs(plain, SIGNATURES, batch_size=1) == SolveStatus.OPTIMAL

    lazy = _NoModelSolver(_solver(INSTANCE))
    stats: dict = dict()
    assert solve_with_lazy_pairs(lazy, SIGNATURES, batch_size=1, stats=stats) == SolveStatus.UNKNOWN
    assert stats["lazyPairsOptimal"] is False
    # Nothing was learned, so nothing was grounded either.
    assert stats["lazyPairsGrounded"] == 0
