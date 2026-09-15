"""`fix_forced_labels` must pin labels the program already implies -- no more, no less.

The soundness claim is that every occurrence `compute_forced_labels` calls good is good in every
model and every one it calls bad is bad in every model. That is checked here directly against
clingo and against the *unmodified* program: for a forced-bad occurrence, adding
``:- not good_action(I,S,A).`` to the program without any forced facts must be unsatisfiable, and
for a forced-good one, adding ``:- good_action(I,S,A).`` must be. Neither test consults the
analysis it is checking, so the check is not circular.
"""

import pytest

from genfond.config_handler import ConfigHandler
from genfond.feature_generator import FeaturePool
from genfond.forced_labels import ActionOccurrence, compute_forced_labels
from genfond.lazy_pairs import solve_with_lazy_pairs
from genfond.solver import Solver, SolveStatus

MAX_COMPLEXITY = 4
SOLVE_PROG = "solve_datalog_sig.lp"


def _occurrence(state, action, signature, candidate=True):
    return ActionOccurrence(0, state, action, signature, candidate)


def _solver(instance, **kwargs):
    return Solver(instance, num_threads=1, solve_prog=SOLVE_PROG, **kwargs)


# --------------------------------------------------------------------------- the four rules


def test_an_action_with_an_unsafe_outcome_is_forced_bad():
    """Step 1: a transition into a state that is neither alive nor pruned can never be good."""
    forced = compute_forced_labels([_occurrence(0, '"a(x)"', 0, candidate=False), _occurrence(0, '"b(x)"', 1)])
    assert {occurrence.action for occurrence in forced.bad} == {'"a(x)"'}
    assert forced.bad_signatures == {0}
    # And with only one candidate left, step 3 forces the other one good.
    assert {occurrence.action for occurrence in forced.good} == {'"b(x)"'}
    assert forced.good_signatures == {1}
    assert not forced.inconsistent


def test_a_bad_class_is_bad_at_every_other_state_too():
    """Step 2, the propagation that crosses states.

    Class 0 is forced bad at state 0 (its outcome is unsafe). At state 1 the same class occurs
    with a perfectly safe outcome, but `:- good_sig(K), bad_sig(K).` still rules it out, which
    leaves state 1 with a single class and forces *that* one good.
    """
    forced = compute_forced_labels(
        [
            _occurrence(0, '"a(x)"', 0, candidate=False),
            _occurrence(0, '"b(x)"', 1),
            _occurrence(1, '"a(y)"', 0),
            _occurrence(1, '"c(y)"', 2),
        ]
    )
    assert {occurrence.action for occurrence in forced.bad} == {'"a(x)"', '"a(y)"'}
    assert forced.good_signatures == {1, 2}
    assert not forced.inconsistent


def test_a_state_whose_candidates_share_one_class_forces_that_class_good():
    """Step 3 with more than one surviving candidate: which one is picked does not matter."""
    forced = compute_forced_labels([_occurrence(0, '"a(x)"', 7), _occurrence(0, '"a(y)"', 7)])
    assert forced.good_signatures == {7}
    assert {occurrence.action for occurrence in forced.good} == {'"a(x)"', '"a(y)"'}


def test_a_good_class_makes_its_occurrences_good_elsewhere():
    """Step 4: state 1 has an alternative, but class 3 is good, so its occurrence there is too."""
    forced = compute_forced_labels(
        [_occurrence(0, '"a(x)"', 3), _occurrence(1, '"a(y)"', 3), _occurrence(1, '"b(y)"', 4)]
    )
    assert forced.good_signatures == {3}
    assert {occurrence.action for occurrence in forced.good} == {'"a(x)"', '"a(y)"'}
    # Nothing forces class 4 either way: the state is already covered by "a(y)".
    assert not forced.bad
    assert not forced.bad_signatures


def test_nothing_is_forced_when_every_state_has_two_independent_options():
    forced = compute_forced_labels([_occurrence(0, '"a(x)"', 0), _occurrence(0, '"b(x)"', 1)])
    assert not forced.good and not forced.bad
    assert not forced.good_signatures and not forced.bad_signatures
    assert not forced.inconsistent


def test_a_state_with_no_candidate_left_is_inconsistent():
    forced = compute_forced_labels([_occurrence(0, '"a(x)"', 0, candidate=False)])
    assert forced.inconsistent


def test_a_class_forced_both_ways_is_inconsistent():
    # Class 0 is forced good at state 0 (sole candidate) and forced bad at state 1 (unsafe).
    forced = compute_forced_labels([_occurrence(0, '"a(x)"', 0), _occurrence(1, '"a(y)"', 0, candidate=False)])
    assert forced.inconsistent
    assert forced.good_signatures & forced.bad_signatures == {0}


# ------------------------------------------------------------ the labels really are implied


@pytest.mark.parametrize("fixture", ["gripper", "blocks_clear"])
def test_every_forced_label_is_implied_by_the_unmodified_program(fixture, request):
    """The soundness check: clingo must refute the opposite of each forced label.

    The instance used here carries no `forced_good`/`forced_bad` facts at all, so what is being
    tested is that the analysis only ever states what the original program already entails.
    """
    domain, problem = request.getfixturevalue(fixture)
    config = ConfigHandler(type="datalog-sig")
    config["fix_forced_labels"] = False
    pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
    instance = pool.to_clingo()
    assert "forced_good" not in instance and "forced_bad" not in instance
    forced = compute_forced_labels(pool._occurrences)
    assert not forced.inconsistent
    assert forced.good or forced.bad, "the fixture forces nothing, so this proves nothing"
    # Otherwise every refutation below would hold vacuously.
    assert _solver(instance).solve(), "the program itself is unsatisfiable"

    # A handful each keeps the test to a few seconds; the ids are deterministic.
    for occurrence in sorted(forced.good)[:6]:
        refuted = _solver(
            instance + f"\n:- good_action({occurrence.instance}, {occurrence.state}, {occurrence.action})."
        )
        assert not refuted.solve(), f"{occurrence} is not actually forced good"
    for occurrence in sorted(forced.bad)[:6]:
        refuted = _solver(
            instance + f"\n:- not good_action({occurrence.instance}, {occurrence.state}, {occurrence.action})."
        )
        assert not refuted.solve(), f"{occurrence} is not actually forced bad"


@pytest.mark.parametrize("fixture", ["gripper", "blocks_clear"])
def test_fixing_the_labels_does_not_change_the_optimum(fixture, request):
    """The equivalence claim: same cost with the flag on and off, eagerly and lazily."""
    domain, problem = request.getfixturevalue(fixture)
    costs = dict()
    for fix in (False, True):
        config = ConfigHandler(type="datalog-sig")
        config["fix_forced_labels"] = fix
        config["lazy_pairs"] = False
        pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
        instance = pool.to_clingo()
        assert ("forced_bad" in instance or "forced_good" in instance) is fix
        eager = _solver(instance)
        assert eager.solve()

        config["lazy_pairs"] = True
        lazy_pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
        lazy = _solver(lazy_pool.to_clingo())
        assert (
            solve_with_lazy_pairs(
                lazy,
                lazy_pool.signatures,
                batch_size=config["lazy_pairs_batch"],
                forced=lazy_pool.forced_labels,
            )
            == SolveStatus.OPTIMAL
        )
        assert lazy.cost[-1] == eager.cost[-1]
        costs[fix] = eager.cost[-1]
    assert costs[True] == costs[False]


def test_the_lazy_loop_seeds_the_pairs_between_two_forced_classes(gripper):
    """Pairs whose two classes both carry a fixed label constrain every model, so they need no
    counterexample and are grounded as batch 0."""
    domain, problem = gripper
    config = ConfigHandler(type="datalog-sig")
    config["fix_forced_labels"] = True
    pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
    solver = _solver(pool.to_clingo())
    stats: dict = dict()
    assert pool.forced_labels is not None
    assert pool.forced_labels.good_signatures and pool.forced_labels.bad_signatures
    status = solve_with_lazy_pairs(
        solver, pool.signatures, batch_size=config["lazy_pairs_batch"], stats=stats, forced=pool.forced_labels
    )
    assert status == SolveStatus.OPTIMAL
    assert stats["lazyPairsSeeded"] > 0
    assert stats["lazyPairsSeededViolated"] >= stats["lazyPairsSeeded"]


def test_the_plan_heuristic_leaves_the_optimum_alone(gripper):
    """`plan_label_heuristic` only reorders the search, so the cost must be identical."""
    domain, problem = gripper
    config = ConfigHandler(type="datalog-sig")
    config["lazy_pairs"] = False
    pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
    plain = _solver(pool.to_clingo())
    assert plain.solve()

    config["plan_label_heuristic"] = True
    guided_pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
    guided = _solver(guided_pool.to_clingo(), plan_label_heuristic=True)
    assert guided.solve()
    assert guided.cost[-1] == plain.cost[-1]


# A state with two same-class actions, one of which lies on an example plan. The heuristic must
# not change which models exist, only the order they are searched in.
PLAN_INSTANCE = """
feature("b_f1"). feature_complexity("b_f1", 1).
state(0, 0). state(0, 1). alive(0, 0). alive(0, 1). goal(0, 1).
trans(0, 0, "a(x0)", 1). trans(0, 0, "a(x1)", 1).
asig(0, 0, "a(x0)", 0). asig(0, 0, "a(x1)", 0).
"""


def test_the_plan_heuristic_grounds_and_keeps_every_model():
    plain = _solver(PLAN_INSTANCE)
    assert plain.solve()
    guided = _solver(PLAN_INSTANCE + '\nplan_action(0, 0, "a(x0)").', plan_label_heuristic=True)
    assert guided.solve()
    assert guided.cost == plain.cost
    # The heuristic biases the good_trans decision towards the on-plan action.
    assert (0, 0, "a(x0)") in guided.solution["good_action"]
