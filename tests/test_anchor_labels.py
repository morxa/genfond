"""Tests for policy-anchored labels (H29).

`policy_conformant_plans` (H27) keeps the best policy P *feasible* on the next round's ASP
instance by recording P's own trajectories as example plans. H29 keeps it *preferred*: every
transition along those trajectories is emitted as ``anchor(I, S, A)`` and constrained to be
labelled good, so a round's model has to agree with P wherever P already worked instead of being
free to pick a cheaper patchwork that happens to fit the enlarged training set.

Three things are checked here:

(a) the constraint itself, against clingo and the raw encoding -- an anchored transition is
    labelled good, its signature class is forced good wherever else it occurs, and an anchor that
    contradicts the graph layer makes the round unsatisfiable;
(b) the fallback -- an unsatisfiable anchored solve is re-solved on the same instance without the
    anchors, and only that second solve is what the round reports (so an anchor can never be read
    as a refutation of the complexity level);
(c) the off switch -- with `anchor_policy_labels` false no `anchor/3` fact is emitted at all.
"""

from typing import Optional, Sequence

from pddl.core import Plan
from pddl.logic import constants

from genfond import iterative_solver as isolver
from genfond.config_handler import ConfigHandler
from genfond.feature_generator import FeaturePool
from genfond.problem_iterator import MAX_COST, Result
from genfond.solver import Solver, SolveStatus

SOLVE_PROG = "solve_datalog_sig.lp"


def _solver(instance, **kwargs):
    return Solver(instance, num_threads=1, solve_prog=SOLVE_PROG, **kwargs)


# --- (a) the constraint ------------------------------------------------------------------------

# One alive non-goal state with two ways to reach a goal, in two different signature classes.
# Nothing in the instance prefers either, so the label is entirely the anchor's doing.
TWO_WAYS = """
feature("b_f1"). feature_complexity("b_f1", 1).
state(0, 0). state(0, 1). state(0, 2).
alive(0, 0). alive(0, 1). alive(0, 2).
goal(0, 1). goal(0, 2).
trans(0, 0, "a(x0)", 1). trans(0, 0, "b(x0)", 2).
asig(0, 0, "a(x0)", 0). asig(0, 0, "b(x0)", 1).
"""


def test_an_anchored_transition_is_labelled_good():
    anchored = _solver(TWO_WAYS + '\nanchor(0, 0, "a(x0)").', anchors=True)
    assert anchored.solve()
    assert (0, 0, "a(x0)") in anchored.solution["good_action"]

    # The other anchor is equally satisfiable, so the first result is the anchor's doing and not
    # some fixed tie-break inside clingo.
    other = _solver(TWO_WAYS + '\nanchor(0, 0, "b(x0)").', anchors=True)
    assert other.solve()
    assert (0, 0, "b(x0)") in other.solution["good_action"]


# The anchored action's only outcome is a state that is neither alive nor pruned, which
# `:- good_trans(I,S,A,_), trans(I,S,A,S2), not alive(I,S2), not pruned(I,S2).` forbids.
# The other action reaches a goal, so the round is perfectly solvable without the anchor.
CONTRADICTING = """
feature("b_f1"). feature_complexity("b_f1", 1).
state(0, 0). state(0, 1). state(0, 2).
alive(0, 0). alive(0, 1).
goal(0, 1).
trans(0, 0, "a(x0)", 2). trans(0, 0, "b(x0)", 1).
asig(0, 0, "a(x0)", 0). asig(0, 0, "b(x0)", 1).
anchor(0, 0, "a(x0)").
"""


def test_a_contradicting_anchor_is_unsatisfiable_and_the_same_instance_solves_without_it():
    anchored = _solver(CONTRADICTING, anchors=True)
    assert not anchored.solve()
    assert anchored.status == SolveStatus.UNSATISFIABLE

    fallback = _solver(CONTRADICTING)
    assert fallback.solve()
    assert (0, 0, "b(x0)") in fallback.solution["good_action"]


# The anchored occurrence and a second one at another state share signature class 0. Goodness is
# a function of the class (`:- good_sig(K), bad_sig(K).`), so the anchor reaches across states.
SHARED_CLASS = """
feature("b_f1"). feature_complexity("b_f1", 1).
state(0, 0). state(0, 1). state(0, 2). state(0, 3). state(0, 4).
alive(0, 0). alive(0, 1). alive(0, 2). alive(0, 3). alive(0, 4).
goal(0, 2). goal(0, 3). goal(0, 4).
trans(0, 0, "a(x0)", 2). trans(0, 0, "b(x0)", 3).
trans(0, 1, "a(x1)", 4). trans(0, 1, "b(x1)", 3).
asig(0, 0, "a(x0)", 0). asig(0, 0, "b(x0)", 1).
asig(0, 1, "a(x1)", 0). asig(0, 1, "b(x1)", 1).
"""


def test_anchoring_one_occurrence_forces_its_signature_class_good_everywhere():
    """This is why anchors need a fallback: the class, not the occurrence, is what gets pinned --
    exactly the cross-state consequence `forced_labels` derives as its step 4."""
    anchored = _solver(SHARED_CLASS + '\nanchor(0, 0, "a(x0)").', anchors=True)
    assert anchored.solve()
    assert (0, 0, "a(x0)") in anchored.solution["good_action"]
    # Not anchored itself, but its class is good, so it cannot be bad anywhere.
    assert (0, 1, "a(x1)") in anchored.solution["good_action"]


# The unquotiented encoding carries the same constraint. Two differently named actions, so the
# pairwise separation constraint is satisfied without selecting anything.
PLAIN_DATALOG = """
feature("b_f1"). feature_complexity("b_f1", 1).
state(0, 0). state(0, 1). state(0, 2).
alive(0, 0). alive(0, 1). alive(0, 2).
goal(0, 1). goal(0, 2).
trans(0, 0, "a(x0)", 1). trans(0, 0, "b(x0)", 2).
aname("a(x0)", "a"). aname("b(x0)", "b").
anchor(0, 0, "a(x0)").
"""


def test_the_anchor_constraint_grounds_in_the_unquotiented_encoding_too():
    anchored = Solver(PLAIN_DATALOG, num_threads=1, solve_prog="solve_datalog.lp", anchors=True)
    assert anchored.solve()
    assert (0, 0, "a(x0)") in anchored.solution["good_action"]


# --- (b) the fallback --------------------------------------------------------------------------


def _datalog_config(**overrides):
    config = ConfigHandler(type="datalog-sig")
    config["lazy_pairs"] = False
    config["num_threads"] = 1
    config["anchor_policy_labels"] = True
    config.update(overrides)
    return config


def _simple_blocks_plan():
    a, b, c = constants("a b c")
    return Plan([("pick", [a, b]), ("put", [a, c])])


class _UnsatisfiableWhenAnchored(Solver):
    """A `Solver` whose anchored solves always come back UNSATISFIABLE.

    Constructing a genuinely contradictory anchor out of a real fixture would mean finding a
    signature collision in it; what has to be tested here is the *caller's* reaction, so the
    contradiction is injected instead. Unanchored solves run for real.
    """

    calls: list[bool] = []

    def __init__(self, *args, **kwargs):
        self._anchored = bool(kwargs.get("anchors", False))
        type(self).calls.append(self._anchored)
        super().__init__(*args, **kwargs)

    def solve(self, bound: Optional[Sequence[int]] = None) -> bool:
        if self._anchored:
            self.status = SolveStatus.UNSATISFIABLE
            self.optimal = True
            return False
        return super().solve(bound=bound)


def test_an_unsatisfiable_anchored_solve_is_retried_without_anchors(monkeypatch, simple_blocks, caplog):
    """(b) The round's answer always comes from a solve without anchors, so an anchored UNSAT can
    never reach `ProblemIterator.set_last_result` as `Result.NO_SOLUTION` and refute a complexity
    level."""
    domain, problem = simple_blocks
    plan = _simple_blocks_plan()
    _UnsatisfiableWhenAnchored.calls = []
    monkeypatch.setattr(isolver, "Solver", _UnsatisfiableWhenAnchored)
    stats: dict = dict()

    result, policy, frontier = isolver.solve_step(
        domain=domain,
        config=_datalog_config(),
        stats=stats,
        example_plans={problem.name: [plan]},
        active_problems=[problem],
        complexity=3,
        all_features=False,
        max_cost=MAX_COST,
        anchor_plans={problem.name: plan},
    )

    assert _UnsatisfiableWhenAnchored.calls == [True, False], "exactly one anchored and one plain solve"
    assert stats["anchorFallbacks"] == 1
    assert stats["anchoredTransitions"] == 2
    assert "Anchored solve unsatisfiable; retrying without anchors" in caplog.text
    # The round reports the *unanchored* solve: a policy, not a refutation.
    assert result == Result.SUCCESS
    assert policy is not None
    assert frontier == []
    assert stats["solveStatus"] == SolveStatus.OPTIMAL.name


def test_without_a_contradiction_the_round_solves_once(monkeypatch, simple_blocks):
    """The fallback costs a second solve only on the rounds that actually hit it."""
    domain, problem = simple_blocks
    plan = _simple_blocks_plan()

    class _Counting(Solver):
        calls: list[bool] = []

        def __init__(self, *args, **kwargs):
            type(self).calls.append(bool(kwargs.get("anchors", False)))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(isolver, "Solver", _Counting)
    stats: dict = dict()

    result, policy, _ = isolver.solve_step(
        domain=domain,
        config=_datalog_config(),
        stats=stats,
        example_plans={problem.name: [plan]},
        active_problems=[problem],
        complexity=3,
        all_features=False,
        max_cost=MAX_COST,
        anchor_plans={problem.name: plan},
    )

    assert _Counting.calls == [True]
    assert "anchorFallbacks" not in stats
    assert result == Result.SUCCESS
    assert policy is not None
    # The policy the anchored round found agrees with the anchored trajectory by construction.
    assert stats["anchoredTransitions"] == 2


def test_anchors_combine_with_the_lazy_pairs_loop(simple_blocks):
    """The `anchor` part is grounded once, at construction, so it holds across every solve of the
    lazy loop -- and the fallback simply builds a second Solver and runs the loop again."""
    domain, problem = simple_blocks
    plan = _simple_blocks_plan()
    stats: dict = dict()

    result, policy, _ = isolver.solve_step(
        domain=domain,
        config=_datalog_config(lazy_pairs=True),
        stats=stats,
        example_plans={problem.name: [plan]},
        active_problems=[problem],
        complexity=3,
        all_features=False,
        max_cost=MAX_COST,
        anchor_plans={problem.name: plan},
    )

    assert result == Result.SUCCESS
    assert policy is not None
    assert stats["anchoredTransitions"] == 2


# --- (c) the off switch, and what is anchored --------------------------------------------------


def test_the_anchored_transitions_are_the_trajectory_s_own_steps(simple_blocks):
    domain, problem = simple_blocks
    plan = _simple_blocks_plan()
    config = _datalog_config()

    pool = FeaturePool(
        domain,
        [problem],
        config,
        max_complexity=2,
        plans={problem.name: [plan]},
        anchor_plans={problem.name: plan},
    )
    instance = pool.to_clingo()

    assert [action for _, _, action in pool.anchored_transitions] == ['"pick(a,b)"', '"put(a,c)"']
    assert pool.anchored_problems == {problem.name}
    assert pool.anchors_dropped == 0
    for instance_id, state_id, action in pool.anchored_transitions:
        assert f"anchor({instance_id}, {state_id}, {action})." in instance
        # An anchor always names a transition the instance actually has.
        assert f"trans({instance_id}, {state_id}, {action}," in instance


def test_anchor_policy_labels_off_emits_no_anchor_facts(simple_blocks):
    """(c) With the option off the instance is exactly the one built before this mechanism."""
    domain, problem = simple_blocks
    plan = _simple_blocks_plan()
    config = _datalog_config(anchor_policy_labels=False)

    pool = FeaturePool(
        domain,
        [problem],
        config,
        max_complexity=2,
        plans={problem.name: [plan]},
        anchor_plans={problem.name: plan},
    )

    assert "anchor(" not in pool.to_clingo()
    assert pool.anchored_transitions == []


def test_a_trajectory_for_a_problem_outside_the_training_set_is_ignored(simple_blocks):
    """Anchors only mean something for problems whose states are in this instance."""
    domain, problem = simple_blocks
    plan = _simple_blocks_plan()

    pool = FeaturePool(
        domain,
        [problem],
        _datalog_config(),
        max_complexity=2,
        plans={problem.name: [plan]},
        anchor_plans={"some-other-problem": plan},
    )

    assert "anchor(" not in pool.to_clingo()
    assert pool.anchored_transitions == []


def test_steps_missing_from_the_plan_restricted_state_space_are_dropped(simple_blocks):
    """A trajectory that leaves the plan-restricted state space is anchored as far as the graph
    goes and skipped after that, instead of pinning a label the instance cannot express."""
    domain, problem = simple_blocks
    a, b, c = constants("a b c")
    # put(a,a) is off every example plan, so its successor is never expanded (it is PRUNED with
    # frontier_expansion on), and the step the trajectory takes *from* there has nothing to
    # anchor to.
    off_plan = Plan([("pick", [a, b]), ("put", [a, a]), ("pick", [a, a])])

    pool = FeaturePool(
        domain,
        [problem],
        _datalog_config(),
        max_complexity=2,
        plans={problem.name: [_simple_blocks_plan()]},
        anchor_plans={problem.name: off_plan},
    )
    pool.to_clingo()

    assert [action for _, _, action in pool.anchored_transitions] == ['"pick(a,b)"', '"put(a,a)"']
    assert pool.anchors_dropped >= 1


# --- which trajectories reach the round, end to end --------------------------------------------


class _DummyPolicy:
    def __init__(self, cost=(2,)):
        self.cost = cost


def _loop_config(**overrides):
    config = {
        "min_complexity": 2,
        "max_complexity": 6,
        "use_unrestricted_features": False,
        "reset_complexity_on_state_space_change": False,
        "add_problem_after_success": False,
        "unselect_problems": False,
        "min_number_of_plans": 1,
        "max_plans_per_problem": None,
        "max_frontier_expansions": 20,
        "use_example_plans": True,
        "frontier_expansion": False,
        "policy_conformant_plans": True,
        "anchor_policy_labels": True,
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
    config.update(overrides)
    return config


def _second_problem(domain):
    """A second problem for the `simple_blocks` domain that the mocked policy cannot solve."""
    from pddl.core import Problem
    from pddl.logic import Predicate, variables

    x, y = variables("x y")
    on = Predicate("on", x, y)
    a, b, c = constants("a b c")
    return Problem(
        "p2",
        domain=domain,
        requirements=domain.requirements,
        objects=[a, b, c],
        init=[on(b, a)],
        goal=on(b, c),
    )


def test_only_the_best_policy_s_trajectories_are_anchored_and_only_on_active_problems(monkeypatch, simple_blocks):
    """The first round has no best policy and anchors nothing. Once one exists, its trajectories
    are handed to the next round -- restricted to the problems that round actually trains on, and
    holding only the problems that policy solved."""
    domain, problem = simple_blocks
    other = _second_problem(domain)
    trajectory = _simple_blocks_plan().instantiate(domain)
    seen_anchor_plans: list = []

    monkeypatch.setattr(isolver, "_get_example_plan_computer", lambda config: (lambda d, p, c: iter([]), {}, "none"))

    def fake_solve_step(**kwargs):
        seen_anchor_plans.append(kwargs["anchor_plans"])
        return Result.SUCCESS, _DummyPolicy(), []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)

    def fake_execute_policy(domain, problem_, policy, config, time_limit=None, out_actions=None):
        if problem_.name != problem.name:
            raise RuntimeError("the policy does not solve p2")
        if out_actions is not None:
            out_actions.extend(trajectory)
        return [str(action) for action in trajectory]

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    isolver.solve_iteratively(domain, [problem, other], _loop_config(stop_after_first_solution=False))

    assert seen_anchor_plans, "the loop must have run at least one round"
    # Round 1: nothing has been validated yet, so there is no best policy to anchor on.
    assert seen_anchor_plans[0] == {}
    # Every later round anchors exactly the best policy's trajectory, on the solved problem only.
    assert len(seen_anchor_plans) > 1
    for anchor_plans in seen_anchor_plans[1:]:
        assert set(anchor_plans) == {problem.name}


def test_anchor_policy_labels_off_passes_no_anchor_plans(monkeypatch, simple_blocks):
    """(c) again, at the loop level: with the option off `solve_step` is called exactly as it was
    before this mechanism existed."""
    domain, problem = simple_blocks
    trajectory = _simple_blocks_plan().instantiate(domain)
    seen_anchor_plans: list = []

    monkeypatch.setattr(isolver, "_get_example_plan_computer", lambda config: (lambda d, p, c: iter([]), {}, "none"))

    def fake_solve_step(**kwargs):
        seen_anchor_plans.append(kwargs["anchor_plans"])
        return Result.SUCCESS, _DummyPolicy(), []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)

    def fake_execute_policy(domain, problem_, policy, config, time_limit=None, out_actions=None):
        if out_actions is not None:
            out_actions.extend(trajectory)
        return [str(action) for action in trajectory]

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    isolver.solve_iteratively(domain, [problem], _loop_config(anchor_policy_labels=False))

    assert seen_anchor_plans == [None]
