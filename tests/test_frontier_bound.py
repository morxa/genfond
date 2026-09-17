"""The frontier lower bound (H31): a proven frontier count settles the whole round.

`solve_datalog_sig.lp` minimises the frontier-transition count at the *highest* priority, and the
lazy loop's grounded program always carries a subset of the full round's separation constraints.
So a proven-optimal cost whose frontier component is k > 0 says something about every model of
the round, not just about this iteration: none of them has fewer than k frontier transitions, and
`solve_step` accepts only a zero-frontier model as a policy. The round is therefore already
decided, and every further lazy iteration can only refine a model that can never become one.

Checked here:

(a) an instance whose relaxed optimum needs a frontier transition stops after iteration 1, and
    the model it returns still reaches `solve_step` as `Result.FRONTIER`;
(b) an instance with `pruned/2` states the optimum does *not* need runs the loop to completion
    exactly as before -- the abort must key on the frontier count, not on the presence of a
    frontier level in the cost vector;
(c) the warm start replays the previous iteration's cost as clingo's initial bound, and gives up
    that bound (rather than reading a refutation into it) when the optimum has risen past it;
(d) an anchored round that proves the bound is re-solved without anchors, because the anchor
    constraint is one the full round does not carry;
(e) a frontier plan dropped at `max_plans_per_problem` is counted and logged, and the states of a
    capped problem are not handed to the planner at all.
"""

import logging
from dataclasses import dataclass

import pytest
from pddl.core import Plan
from pddl.logic import Constant, Predicate, constants

from genfond import iterative_solver as isolver
from genfond.action_signatures import ActionSignature
from genfond.config_handler import ConfigHandler
from genfond.cost_utils import prune_cost
from genfond.lazy_pairs import solve_with_lazy_pairs
from genfond.problem_iterator import MAX_COST, ProblemIterator, Result
from genfond.solver import Solver, SolveStatus

SOLVE_PROG = "solve_datalog_sig.lp"


def _signature(name, arity, concepts, roles, bools):
    return ActionSignature(name, arity, frozenset(concepts), frozenset(roles), tuple(bools))


# Four same-named actions in four signature classes, each cheapest to separate from the first by
# a different element, so a batch size of one forces several lazy iterations. Reused by every
# instance below; only the transition targets differ.
SIGNATURES = [
    _signature("a", 1, [], [], [("b_f1", 1), ("b_f2", 1), ("b_f3", 1)]),
    _signature("a", 1, [], [], [("b_f1", 0), ("b_f2", 1), ("b_f3", 1)]),
    _signature("a", 1, [], [], [("b_f1", 1), ("b_f2", 0), ("b_f3", 1)]),
    _signature("a", 1, [("c_c1", 0)], [], [("b_f1", 1), ("b_f2", 1), ("b_f3", 1)]),
]

ELEMENTS = """
feature("b_f1"). feature_complexity("b_f1", 1).
feature("b_f2"). feature_complexity("b_f2", 2).
feature("b_f3"). feature_complexity("b_f3", 9).
concept("c_c1"). concept_complexity("c_c1", 4).
"""

CLASSES = """
asig(0, 0, "a(x0)", 0). asig(0, 0, "a(x1)", 1).
asig(0, 0, "a(x2)", 2). asig(0, 0, "a(x3)", 3).
"""

# State 0's only way out is a(x0), which lands on the unexpanded state 2; the goal state 1 is
# unreachable and the other three actions loop back to state 0, which can never be safe on its
# own. So *every* model takes the frontier transition and the relaxed optimum proves it.
FORCED_FRONTIER = (
    ELEMENTS
    + """
state(0, 0). alive(0, 0).
state(0, 1). alive(0, 1). goal(0, 1).
state(0, 2). pruned(0, 2).
trans(0, 0, "a(x0)", 2).
trans(0, 0, "a(x1)", 0). trans(0, 0, "a(x2)", 0). trans(0, 0, "a(x3)", 0).
"""
    + CLASSES
)

# The same instance with a(x0) reaching the goal instead. The frontier level still grounds (state
# 2 is pruned and reachable), but taking a(x1..3) into it costs more than separating them away,
# so the optimum has a frontier count of zero and the loop must run to the end.
OPTIONAL_FRONTIER = (
    ELEMENTS
    + """
state(0, 0). alive(0, 0).
state(0, 1). alive(0, 1). goal(0, 1).
state(0, 2). pruned(0, 2).
trans(0, 0, "a(x0)", 1).
trans(0, 0, "a(x1)", 2). trans(0, 0, "a(x2)", 2). trans(0, 0, "a(x3)", 2).
"""
    + CLASSES
)


def _solver(instance, **kwargs):
    return Solver(instance, num_threads=1, solve_prog=SOLVE_PROG, **kwargs)


# --- cost_utils.prune_cost ----------------------------------------------------------------------


def test_prune_cost_reads_the_frontier_level_only_when_it_grounded():
    # Two levels: frontier first, complexity last (the cost-vector gotcha in AGENTS.md).
    assert prune_cost([3, 7]) == 3
    # One level: the frontier minimize never grounded, so cost[0] is the complexity.
    assert prune_cost([7]) == 0
    assert prune_cost([]) == 0
    # An "above" bias inserts a level between the two; "below" ones push complexity off the end.
    assert prune_cost([3, 2, 7], minimize_good_signatures="above") == 3
    assert prune_cost([2, 7], minimize_good_signatures="above") == 0
    assert prune_cost([3, 7, 2], minimize_selected_count="below") == 3
    assert prune_cost([7, 2], minimize_selected_count="below") == 0


def test_prune_count_agrees_with_the_model_s_own_frontier_atoms():
    """The positional read is only as good as its agreement with what the model actually did."""
    forced = _solver(FORCED_FRONTIER)
    assert forced.solve()
    assert forced.prune_count == len(forced.solution["frontier"]) == 1

    optional = _solver(OPTIONAL_FRONTIER)
    assert optional.solve()
    # The level grounded (pruned/2 is reachable) but the optimum does not use it.
    assert len(optional.cost) == 2
    assert optional.prune_count == 0
    assert "frontier" not in optional.solution


# --- (a) the abort ------------------------------------------------------------------------------


def test_a_proven_frontier_optimum_stops_the_lazy_loop_after_the_first_iteration(caplog):
    caplog.set_level(logging.INFO, logger="genfond.lazy_pairs")
    solver = _solver(FORCED_FRONTIER)
    stats: dict = dict()

    status = solve_with_lazy_pairs(solver, SIGNATURES, batch_size=1, stats=stats)

    assert status == SolveStatus.OPTIMAL
    assert stats["lazyPairIterations"] == 1
    assert stats["frontierLowerBoundAbort"] is True
    assert stats["frontierLowerBound"] == 1
    assert "Frontier lower bound 1 > 0 proven after lazy iteration 1" in caplog.text
    # The model is still handed back: the caller wants its frontier states, nothing else.
    assert solver.solution["frontier"] == {(0, 2)}


def test_without_the_abort_the_same_round_keeps_refining_a_model_that_cannot_be_a_policy():
    """The measurement the abort is for: every one of those iterations is a wasted solve."""
    solver = _solver(FORCED_FRONTIER)
    stats: dict = dict()

    status = solve_with_lazy_pairs(solver, SIGNATURES, batch_size=1, stats=stats, frontier_lower_bound_abort=False)

    assert status == SolveStatus.OPTIMAL
    assert stats["lazyPairIterations"] > 1
    assert stats["frontierLowerBoundAbort"] is False
    # And the extra work bought nothing: the model it converges on still needs the frontier.
    assert solver.prune_count == 1


def test_a_merely_feasible_model_never_triggers_the_abort():
    """A cost that was not *proved* optimal bounds the minimum from above, so it proves nothing."""

    class _CutOff(Solver):
        def solve(self, bound=None):
            found = super().solve(bound=bound)
            if found:
                self.status = SolveStatus.SATISFIABLE
                self.optimal = False
            return found

    solver = _CutOff(FORCED_FRONTIER, num_threads=1, solve_prog=SOLVE_PROG)
    stats: dict = dict()

    status = solve_with_lazy_pairs(solver, SIGNATURES, batch_size=1, stats=stats)

    assert status == SolveStatus.SATISFIABLE
    assert stats["frontierLowerBoundAbort"] is False
    assert stats["lazyPairIterations"] > 1


def test_the_abort_reaches_solve_step_as_a_frontier_result(simple_blocks, monkeypatch):
    """End to end: the aborted round must look exactly like any other frontier round."""
    domain, problem = simple_blocks
    config = ConfigHandler(type="datalog-sig")
    config["num_threads"] = 1
    config["lazy_pairs"] = False

    class _AlwaysFrontier(Solver):
        def solve(self, bound=None):
            found = super().solve(bound=bound)
            # Pretend the round's optimum needs one frontier transition, and that the model says
            # so, without having to build a fixture whose state space really does.
            self.cost = [1] + list(self.cost)
            self.solution["frontier"] = {(0, 0)}
            return found

    monkeypatch.setattr(isolver, "Solver", _AlwaysFrontier)
    stats: dict = dict()

    result, policy, frontier_states = isolver.solve_step(
        domain=domain,
        config=config,
        stats=stats,
        example_plans={problem.name: []},
        active_problems=[problem],
        complexity=2,
        all_features=False,
        max_cost=MAX_COST,
    )

    assert result == Result.FRONTIER
    assert policy is None
    assert frontier_states
    assert stats["frontierLowerBoundAbort"] is True
    assert stats["frontierLowerBound"] == 1


# --- (b) a zero-frontier optimum ----------------------------------------------------------------


def test_a_zero_frontier_optimum_runs_the_loop_to_completion():
    solver = _solver(OPTIONAL_FRONTIER)
    stats: dict = dict()

    status = solve_with_lazy_pairs(solver, SIGNATURES, batch_size=1, stats=stats)

    assert status == SolveStatus.OPTIMAL
    assert stats["lazyPairIterations"] > 1, "the pairs still have to be discovered one at a time"
    assert stats["frontierLowerBoundAbort"] is False
    assert "frontierLowerBound" not in stats
    assert solver.prune_count == 0
    # b_f1 (1) + b_f2 (2) + c_c1 (4); b_f3 is constant across the classes and separates nothing.
    assert solver.feature_complexity == 7


def test_the_abort_changes_nothing_when_no_state_is_pruned():
    """With no `pruned/2` fact the frontier level never grounds, and `cost[0]` is the complexity."""
    program = OPTIONAL_FRONTIER.replace("pruned(0, 2).", "alive(0, 2). goal(0, 2).")
    with_abort = _solver(program)
    stats: dict = dict()
    assert solve_with_lazy_pairs(with_abort, SIGNATURES, batch_size=1, stats=stats) == SolveStatus.OPTIMAL
    assert len(with_abort.cost) == 1
    assert with_abort.prune_count == 0
    assert stats["frontierLowerBoundAbort"] is False


# --- (c) the warm start -------------------------------------------------------------------------


class _RecordingSolver(Solver):
    """A real `Solver` that remembers the warm-start bound each solve was given."""

    def __init__(self, *args, **kwargs):
        self.bounds: list = []
        super().__init__(*args, **kwargs)

    def solve(self, bound=None):
        self.bounds.append(None if bound is None else list(bound))
        return super().solve(bound=bound)


def test_a_warm_start_bound_is_passed_to_every_iteration_after_the_first():
    solver = _RecordingSolver(OPTIONAL_FRONTIER, num_threads=1, solve_prog=SOLVE_PROG)
    stats: dict = dict()

    assert solve_with_lazy_pairs(solver, SIGNATURES, batch_size=1, stats=stats) == SolveStatus.OPTIMAL

    assert solver.bounds[0] is None, "there is nothing to warm-start the first solve from"
    assert any(bound is not None for bound in solver.bounds[1:])


def test_warm_starting_does_not_change_the_answer():
    cold = _solver(OPTIONAL_FRONTIER)
    assert solve_with_lazy_pairs(cold, SIGNATURES, batch_size=1, warm_start=False) == SolveStatus.OPTIMAL
    warm = _solver(OPTIONAL_FRONTIER)
    assert solve_with_lazy_pairs(warm, SIGNATURES, batch_size=1, warm_start=True) == SolveStatus.OPTIMAL
    assert warm.cost == cold.cost


def test_a_solve_is_never_worse_than_the_bound_it_was_given():
    """The property the warm start exists for: an iteration cannot come back with a worse cost."""
    solver = _RecordingSolver(OPTIONAL_FRONTIER, num_threads=1, solve_prog=SOLVE_PROG)
    costs: list = []

    original = solver.solve

    def recording(bound=None):
        found = original(bound=bound)
        costs.append((None if bound is None else list(bound), list(solver.cost)))
        return found

    solver.solve = recording  # type: ignore[method-assign]
    assert solve_with_lazy_pairs(solver, SIGNATURES, batch_size=1) == SolveStatus.OPTIMAL
    for bound, cost in costs:
        if bound is not None and cost:
            assert cost <= bound, "a bounded solve returned a model the bound should have hidden"


def test_a_bound_the_added_pairs_outgrow_is_dropped_rather_than_read_as_a_refutation(caplog):
    """Adding pairs can only *raise* the optimum, so the bound is a restriction, not a constraint.

    The first relaxed solve of `OPTIONAL_FRONTIER` selects nothing at all (no pair constrains it
    yet), so its cost is the cheapest the round can ever be; every later iteration is forced above
    it. A warm bound taken from it is therefore unsatisfiable, and the loop must re-solve without
    it instead of reporting the round unsatisfiable.
    """
    caplog.set_level(logging.INFO, logger="genfond.lazy_pairs")
    solver = _solver(OPTIONAL_FRONTIER)
    stats: dict = dict()

    status = solve_with_lazy_pairs(solver, SIGNATURES, batch_size=1, stats=stats)

    assert status == SolveStatus.OPTIMAL
    assert stats["lazyWarmStartRelaxed"] == 1, "the bound is relaxed at most once per round"
    assert "re-solving without it" in caplog.text
    assert solver.feature_complexity == 7


def test_the_bound_is_inclusive_and_a_tighter_one_is_unsatisfiable():
    """`Solver.solve(bound=...)` semantics, pinned directly against clingo."""
    solver = _solver(OPTIONAL_FRONTIER)
    assert solver.solve()
    optimum = list(solver.cost)

    assert solver.solve(bound=optimum), "the bound must admit a model *of* that cost"
    assert list(solver.cost) == optimum

    tighter = list(optimum)
    tighter[-1] -= 1
    assert not solver.solve(bound=tighter)
    assert solver.status == SolveStatus.UNSATISFIABLE
    # ... and dropping the bound brings the model straight back, which is what makes the loop's
    # re-solve sound: nothing about the program changed.
    assert solver.solve()
    assert list(solver.cost) == optimum


# --- (d) the anchor fallback --------------------------------------------------------------------


def test_an_anchored_frontier_lower_bound_is_retried_without_anchors(monkeypatch, simple_blocks, caplog):
    """`:- anchor(I,S,A), not good_action(I,S,A)` is a constraint the full round does not carry.

    So "every model needs a frontier transition" proven under anchors is a statement about the
    anchored program alone, and the round's answer has to come from a solve without them -- the
    same reason an anchored UNSAT is retried.
    """
    caplog.set_level(logging.INFO, logger="genfond.iterative_solver")
    domain, problem = simple_blocks
    a, b, c = constants("a b c")
    plan = Plan([("pick", [a, b]), ("put", [a, c])])
    config = ConfigHandler(type="datalog-sig")
    config["num_threads"] = 1
    config["lazy_pairs"] = False
    config["anchor_policy_labels"] = True

    class _FrontierBoundWhenAnchored(Solver):
        calls: list[bool] = []

        def __init__(self, *args, **kwargs):
            self._anchored = bool(kwargs.get("anchors", False))
            type(self).calls.append(self._anchored)
            super().__init__(*args, **kwargs)

        def solve(self, bound=None):
            found = super().solve(bound=bound)
            if self._anchored and found:
                # As if the anchors had left the solver no way out but the frontier.
                self.cost = [2] + list(self.cost)
            return found

    _FrontierBoundWhenAnchored.calls = []
    monkeypatch.setattr(isolver, "Solver", _FrontierBoundWhenAnchored)
    stats: dict = dict()

    result, policy, frontier_states = isolver.solve_step(
        domain=domain,
        config=config,
        stats=stats,
        example_plans={problem.name: [plan]},
        active_problems=[problem],
        complexity=3,
        all_features=False,
        max_cost=MAX_COST,
        anchor_plans={problem.name: plan},
    )

    assert _FrontierBoundWhenAnchored.calls == [True, False], "exactly one anchored and one plain solve"
    assert stats["anchorFallbacks"] == 1
    assert "needs a frontier transition in every model" in caplog.text
    # The unanchored solve decides, and it finds a real policy.
    assert result == Result.SUCCESS
    assert policy is not None
    assert frontier_states == []
    # The anchored attempt's verdict must not be left standing in the round's stats.
    assert stats["frontierLowerBoundAbort"] is False


def test_the_fallback_is_not_taken_when_the_anchored_round_needs_no_frontier(monkeypatch, simple_blocks):
    """The widened condition must not cost a second solve on every anchored round."""
    domain, problem = simple_blocks
    a, b, c = constants("a b c")
    plan = Plan([("pick", [a, b]), ("put", [a, c])])
    config = ConfigHandler(type="datalog-sig")
    config["num_threads"] = 1
    config["lazy_pairs"] = False
    config["anchor_policy_labels"] = True

    class _Counting(Solver):
        calls: list[bool] = []

        def __init__(self, *args, **kwargs):
            type(self).calls.append(bool(kwargs.get("anchors", False)))
            super().__init__(*args, **kwargs)

    _Counting.calls = []
    monkeypatch.setattr(isolver, "Solver", _Counting)

    result, policy, _ = isolver.solve_step(
        domain=domain,
        config=config,
        stats={},
        example_plans={problem.name: [plan]},
        active_problems=[problem],
        complexity=3,
        all_features=False,
        max_cost=MAX_COST,
        anchor_plans={problem.name: plan},
    )

    assert _Counting.calls == [True]
    assert result == Result.SUCCESS
    assert policy is not None


# --- (e) capped frontier plans ------------------------------------------------------------------


@dataclass(frozen=True)
class DummyProblem:
    name: str
    init: frozenset


def a_plan(*names):
    return Plan([(n, [Constant("x")]) for n in names])


def _iterator(**overrides):
    config = {
        "min_complexity": 2,
        "max_complexity": 7,
        "use_selected_states": False,
        "use_unrestricted_features": False,
        "unselect_problems": False,
        "min_number_of_plans": 1,
        "max_frontier_expansions": 20,
        "reset_complexity_on_state_space_change": False,
        "add_problem_after_success": False,
        "max_plans_per_problem": 1,
    }
    config.update(overrides)
    a, b = constants("a b")
    problems = [
        DummyProblem("p1", frozenset({Predicate("at", a)})),
        DummyProblem("p2", frozenset({Predicate("at", b)})),
    ]
    plans = {p.name: iter([a_plan(f"{p.name}-{i}") for i in range(10)]) for p in problems}
    iterator = iter(ProblemIterator(problems, config, plans=plans))
    next(iterator)
    return iterator


def test_frontier_plans_dropped_at_the_cap_are_counted_and_logged(caplog):
    caplog.set_level(logging.INFO, logger="genfond.problem_iterator")
    iterator = _iterator()
    # min_number_of_plans=1 already put one plan into p1, which is the whole cap.
    assert len(iterator.active_plans["p1"]) == 1

    iterator.record_frontier_expansion({"p1": [a_plan("f1"), a_plan("f2"), a_plan("f3")]}, {})

    assert iterator.frontier_plans_dropped == 3
    assert iterator.frontier_progress is False, "nothing was learned, so the retry must not fire"
    assert len(iterator.active_plans["p1"]) == 1
    assert "Dropping 3 frontier plan(s) for p1" in caplog.text


def test_only_the_plans_past_the_cap_are_counted_as_dropped():
    iterator = _iterator(max_plans_per_problem=2)
    assert len(iterator.active_plans["p1"]) == 1

    iterator.record_frontier_expansion({"p1": [a_plan("f1"), a_plan("f2"), a_plan("f3")]}, {})

    assert len(iterator.active_plans["p1"]) == 2
    assert iterator.frontier_plans_dropped == 2
    assert iterator.frontier_progress is True


def test_nothing_is_counted_without_a_cap():
    iterator = _iterator(max_plans_per_problem=None)
    iterator.record_frontier_expansion({"p1": [a_plan("f1"), a_plan("f2")]}, {})
    assert iterator.frontier_plans_dropped == 0
    assert len(iterator.active_plans["p1"]) == 3


def test_plan_cap_reached_lets_the_caller_skip_the_planner_call():
    """`solve_iteratively` filters the frontier states by this before calling `expand_frontier`."""
    iterator = _iterator()
    assert iterator.plan_cap_reached("p1") is True
    # p2 is not in the training set yet and holds no plans at all.
    assert iterator.plan_cap_reached("p2") is False


def test_a_capped_problem_s_frontier_states_never_reach_the_planner():
    """The saving itself: `solve_iteratively` filters, so `expand_frontier` gets nothing to do."""
    from genfond.frontier import FrontierState, expand_frontier

    states = [FrontierState(problem_name="p1", node_id=i, state=frozenset(), prefix=()) for i in range(2)]

    capped = _iterator(max_plans_per_problem=1)
    assert [s for s in states if not capped.plan_cap_reached(s.problem_name)] == []

    uncapped = _iterator(max_plans_per_problem=None)
    assert [s for s in states if not uncapped.plan_cap_reached(s.problem_name)] == states

    calls: list = []

    def planner(domain_str, problem_str, planner_config):
        calls.append(problem_str)
        return iter([])

    # An empty state list is what the filter hands over for the capped problem, and it costs the
    # planner nothing -- no re-rooting, no call.
    new_plans, dead = expand_frontier(None, {}, [], planner, {}, {"max_frontier_states_per_round": 0})
    assert calls == [] and new_plans == {} and dead == {}
