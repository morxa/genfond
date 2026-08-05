"""Raw-ASP tests for frontier transitions in the datalog encoding.

A `pruned/2` state is an unexpanded frontier state: the solver may route through it on the
assumption that it is solvable, but pays for it at a higher optimization priority than feature
complexity, and reports it as `frontier/2` so the caller can expand it.
"""

from genfond.iterative_solver import max_prune_cost
from genfond.problem_iterator import MAX_COST
from genfond.solver import Solver

# The only route to the goal runs through the unexpanded state 1.
FORCED_FRONTIER = """
    feature("b_f").
    feature_complexity("b_f", 1).

    state(0, 0).
    alive(0, 0).
    eval(0, 0, "b_f", 0).

    state(0, 1).
    pruned(0, 1).

    trans(0, 0, "a(x)", 1).
    aname("a(x)", "a").
    aparam("a(x)", 0, "x").
"""

# State 0 can either take both same-named actions (no separation needed, but a(y) lands on the
# unexpanded state 2), or take only a(x) to the goal, which then requires an expensive concept
# to separate a(x) from the now non-good a(y).
FRONTIER_VS_EXPENSIVE_CONCEPT = """
    concept("c_exp").
    concept_complexity("c_exp", 10).

    state(0, 0).
    alive(0, 0).

    state(0, 1).
    alive(0, 1).
    goal(0, 1).

    state(0, 2).
    pruned(0, 2).

    trans(0, 0, "a(x)", 1).
    aname("a(x)", "a").
    aparam("a(x)", 0, "x").

    trans(0, 0, "a(y)", 2).
    aname("a(y)", "a").
    aparam("a(y)", 0, "y").

    c_eval(0, 0, "c_exp", "x").
"""


def datalog_solver(program: str, **kwargs) -> Solver:
    return Solver(program, solve_prog="solve_datalog.lp", **kwargs)


def test_frontier_transition_is_reported():
    solver = datalog_solver(FORCED_FRONTIER)
    assert solver.solve()
    assert solver.solution["frontier"] == {(0, 1)}
    assert solver.solution["good_action"] == {(0, 0, "a(x)")}
    # Two cost levels: the frontier count comes first, feature complexity last.
    assert len(solver.cost) == 2
    assert solver.cost[0] == 1


def test_no_frontier_atoms_without_pruned_states():
    # Same instance, but state 1 is an ordinary alive goal state.
    program = FORCED_FRONTIER.replace("pruned(0, 1).", "alive(0, 1).\n    goal(0, 1).")
    solver = datalog_solver(program)
    assert solver.solve()
    assert "frontier" not in solver.solution
    # The @2 level does not ground at all when no pruned/2 fact exists, so the cost vector
    # shrinks. Nothing may index cost[0] expecting the feature complexity.
    assert len(solver.cost) == 1


def test_avoiding_the_frontier_outranks_feature_cost():
    solver = datalog_solver(FRONTIER_VS_EXPENSIVE_CONCEPT)
    assert solver.solve()
    # The solver pays 10 in concept complexity rather than use one frontier transition.
    assert solver.cost == [0, 10]
    assert "frontier" not in solver.solution
    assert solver.solution["good_action"] == {(0, 0, "a(x)")}
    assert solver.solution["c_selected"] == {"name", "c_exp"}


def test_frontier_is_used_when_it_is_the_only_option():
    # Drop the concept, so separating a(x) from a(y) becomes impossible and the only model
    # left is the one that takes both actions, routing through the frontier state.
    program = FRONTIER_VS_EXPENSIVE_CONCEPT.replace('concept("c_exp").', "").replace(
        'concept_complexity("c_exp", 10).', ""
    )
    solver = datalog_solver(program)
    assert solver.solve()
    assert solver.solution["frontier"] == {(0, 2)}
    assert solver.cost[0] == 1


def test_limit_prune_cost_forbids_frontier_transitions():
    solver = datalog_solver(FORCED_FRONTIER, max_prune_cost=0)
    assert solver.solve() is False, f"Unexpected solution {solver.solution}"


def test_max_prune_cost_sentinel_is_not_grounded():
    # MAX_COST means "unbounded" and must leave limit_prune_cost ungrounded.
    solver = datalog_solver(FORCED_FRONTIER, max_prune_cost=MAX_COST)
    assert solver.solve()
    assert solver.solution["frontier"] == {(0, 1)}


def test_dead_states_cannot_be_selected():
    # A state with neither alive/2 nor pruned/2 is a refuted dead end: no way out of state 0.
    program = FORCED_FRONTIER.replace("pruned(0, 1).", "")
    solver = datalog_solver(program)
    assert solver.solve() is False, f"Unexpected solution {solver.solution}"


def test_frontier_is_closed_once_a_policy_exists():
    config = {"max_frontier_transitions": None}
    # No policy yet for this configuration: the frontier is available without a cap.
    assert max_prune_cost(config, MAX_COST) == MAX_COST
    # A tightened budget means we already have a policy and are only trying to beat its cost.
    # Expanding the frontier there buys cheaper features, not solvability, so it is forbidden.
    assert max_prune_cost(config, 4) == 0
    assert max_prune_cost({"max_frontier_transitions": 1}, MAX_COST) == 1
    assert max_prune_cost({"max_frontier_transitions": 1}, 4) == 0
