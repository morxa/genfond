"""Tests for policy-conformant example plans (H27).

A round's policy P may solve nearly every problem when *executed* while a policy P* that also
solves the next problem added to the training set is infeasible on the ASP instance: the sampled
SIW plans do not contain the trajectory P* takes, and the plan-restricted `StateSpaceGraph`
therefore lacks those transitions. Recording the trajectory a validated policy actually took, as
an example plan for that problem, removes that dependence on which plans the sample happened to
contain.

The end-to-end tests mock `solve_step` and `execute_policy` (as
`tests/test_in_loop_validation.py` and `tests/test_final_cost_minimization.py` do) but run the
real `ProblemIterator` and the real `PlanStateCoverage` over the in-memory `simple_blocks`
fixture, so plan replay and deduplication are genuinely exercised.
"""

from pddl.core import Plan
from pddl.logic import Constant, Predicate, constants

from genfond import iterative_solver as isolver
from genfond.config_handler import ConfigHandler
from genfond.datalog_policy import DatalogPolicy, DatalogPolicyRule
from genfond.execute_datalog_policy import execute_datalog_policy
from genfond.problem_iterator import ProblemIterator, Result, plan_from_actions, plan_key
from genfond.state_space_generator import Alive, StateSpaceGraph, check_formula, plan_visited_states

# --- (a) an executed policy yields a root-anchored plan that replays to a goal ----------------


def test_executed_policy_trajectory_is_a_root_anchored_plan_that_reaches_the_goal(blocks_clear):
    """`out_actions` turns one policy execution into a `Plan` in exactly the representation every
    other plan producer uses, and that plan replays from `problem.init` to a goal state."""
    domain, problem = blocks_clear
    policy = DatalogPolicy(
        [
            DatalogPolicyRule("unstack(X, Y)", concepts=[("X", "c_primitive(clear, 0)")]),
            DatalogPolicyRule("putdown(X)", concepts=[("X", "c_primitive(clear, 0)")]),
        ]
    )
    taken: list = []

    action_strings = execute_datalog_policy(domain, problem, policy, ConfigHandler(), out_actions=taken)

    # The trajectory is the same sequence the return value describes, as ground actions.
    assert len(taken) == len(action_strings)
    assert taken, "the goal does not hold initially, so the policy must have applied something"

    plan = plan_from_actions(taken)
    # Root-anchored: replaying it from problem.init reaches a state satisfying the goal.
    visited = plan_visited_states(domain, problem, plan)
    assert problem.init in visited
    assert any(check_formula(state, problem.goal) for state in visited)

    # And it replays inside the plan-restricted StateSpaceGraph, which is what the solver sees.
    graph = StateSpaceGraph(domain, problem, plans=[plan])
    assert graph.root.alive == Alive.ALIVE
    assert any(node.goal for node in graph.nodes.values())


# --- ProblemIterator-level unit tests ---------------------------------------------------------


class DummyProblem:
    def __init__(self, name, objects=()):
        self.name = name
        self.objects = objects


def a_plan(*names):
    return Plan([(n, [Constant("x")]) for n in names])


def iterator_config(**overrides):
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
    }
    config.update(overrides)
    return config


def a_started_iterator(config, num_plans=10):
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    plans = {p.name: iter([a_plan(f"{p.name}-{i}") for i in range(num_plans)]) for p in problems}
    iterator = iter(ProblemIterator(problems, config, plans=plans))
    return iterator, next(iterator)


def test_policy_plans_are_exempt_from_max_plans_per_problem():
    """The cap exists to stop runaway growth from INC_PLANS/frontier expansion. A trajectory a
    working policy took is the most valuable plan there is, so it is added in front and does not
    count against the cap -- capping it away is exactly what reintroduces the infeasibility this
    mechanism removes."""
    iterator, _ = a_started_iterator(iterator_config(max_plans_per_problem=1))
    assert len(iterator.active_plans["p1"]) == 1  # the min_number_of_plans floor
    assert iterator._plan_cap_reached("p1") is True

    added, affected = iterator.record_policy_plans({"p1": a_plan("policy-step")})

    assert (added, affected) == (1, 1)
    # In front of the planner's own plans, and still not counted as capped.
    assert plan_key(iterator.active_plans["p1"][0]) == plan_key(a_plan("policy-step"))
    assert len(iterator.active_plans["p1"]) == 2
    assert iterator.policy_plan_counts["p1"] == 1
    assert iterator._plan_cap_reached("p1") is True  # still exactly one *capped* plan


def test_policy_plans_survive_the_problem_being_added_to_the_training_set():
    """A plan recorded for a problem that is not in the training set yet is the whole point: it
    must not be wiped when `_add_next_problem` seeds that problem's plan list, and it must not
    eat into the `min_number_of_plans` floor either."""
    iterator, _ = a_started_iterator(iterator_config(min_number_of_plans=2))
    iterator.record_policy_plans({"p2": a_plan("policy-step")})
    assert [p.name for p in iterator.active_problems] == ["p1"]

    iterator.set_last_result(Result.SUCCESS, cost=(3,))
    iterator.set_solved(iterator.problems[0], True)
    next(iterator)  # p2 joins the training set

    assert [p.name for p in iterator.active_problems] == ["p1", "p2"]
    assert plan_key(iterator.active_plans["p2"][0]) == plan_key(a_plan("policy-step"))
    # The floor of 2 planner plans is drawn on top of the policy plan, not reduced by it.
    assert len(iterator.active_plans["p2"]) == 3


def test_duplicate_and_empty_policy_plans_are_ignored():
    iterator, _ = a_started_iterator(iterator_config())
    assert iterator.record_policy_plans({"p1": a_plan("policy-step")}) == (1, 1)
    # The same trajectory again on a later round changes nothing.
    assert iterator.record_policy_plans({"p1": a_plan("policy-step")}) == (0, 0)
    # A problem whose goal already held initially yields an empty trajectory, which is useless.
    assert iterator.record_policy_plans({"p2": Plan([])}) == (0, 0)


def climb_to_complexity_3(iterator):
    """Refute complexity 2, which pushes the sweep to 3 (the ladder's INC_COMPLEXITY branch)."""
    iterator.set_last_result(Result.NO_SOLUTION)
    round_kwargs = next(iterator)
    assert round_kwargs["complexity"] == 3
    return round_kwargs


def test_policy_plans_invalidate_the_refuted_complexity_level():
    """The bound must be dropped exactly as INC_PLANS and EXPAND_FRONTIER drop it: the
    refutation was established over a state space that the new plan just enlarged, and a simpler
    policy may have become expressible in it."""
    iterator, _ = a_started_iterator(iterator_config())
    climb_to_complexity_3(iterator)
    iterator.set_last_result(Result.NO_SOLUTION)
    assert iterator.refuted_complexity == 3
    assert iterator.enforce_highest_complexity() is True

    iterator.record_policy_plans({"p1": a_plan("policy-step")})

    assert iterator.refuted_complexity == iterator.config["min_complexity"] - 1
    assert iterator.enforce_highest_complexity() is False


def test_policy_plans_block_the_add_a_problem_branch_from_carrying_its_bound_over():
    """`_add_next_problem` normally resumes at `succ_complexity` with `refuted_complexity =
    succ_complexity - 1`, because adding an *instance* is monotone. Enlarging an existing
    instance's state space is not covered by that argument, so a policy plan recorded after the
    success must drop the bound."""
    iterator, _ = a_started_iterator(iterator_config())
    climb_to_complexity_3(iterator)
    # cost 4 -> max_cost 3, which stops the ladder from climbing complexity again and takes the
    # add-a-problem branch instead.
    iterator.set_last_result(Result.SUCCESS, cost=(4,))
    iterator.set_solved(iterator.problems[0], True)
    assert iterator.succ_complexity == 3
    assert iterator.plans_added_since_success is False

    iterator.record_policy_plans({"p1": a_plan("policy-step")})
    assert iterator.plans_added_since_success is True
    next(iterator)  # the add-a-problem branch

    assert [p.name for p in iterator.active_problems] == ["p1", "p2"]
    # Without the guard this would resume at succ_complexity - 1 == 2 and enforce a feature of
    # complexity >= 3, excluding any simpler policy the new plan made expressible.
    assert iterator.refuted_complexity == iterator.config["min_complexity"] - 1
    assert iterator.enforce_highest_complexity() is False


# --- (b)/(c) end to end through solve_iteratively ---------------------------------------------


class DummyPolicy:
    def __init__(self, cost=(2,)):
        self.cost = cost


def solver_config(**overrides):
    config = {
        "min_complexity": 2,
        "max_complexity": 6,
        "use_unrestricted_features": False,
        "reset_complexity_on_state_space_change": False,
        "add_problem_after_success": False,
        "unselect_problems": False,
        "min_number_of_plans": 5,
        "max_plans_per_problem": None,
        "max_frontier_expansions": 20,
        "use_example_plans": True,
        "frontier_expansion": False,
        "policy_conformant_plans": True,
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


def run_one_round(monkeypatch, simple_blocks, **config_overrides):
    """One successful mocked round over the `simple_blocks` fixture; returns the live iterator.

    The mocked policy "solves" the problem by walking pick(a,b), put(a,c) -- a genuine plan for
    this problem, so `PlanStateCoverage` and `StateSpaceGraph` see something real.
    """
    domain, problem = simple_blocks
    a, b, c = constants("a b c")
    trajectory = Plan([("pick", [a, b]), ("put", [a, c])]).instantiate(domain)

    # No planner: the example-plan iterators are empty, so every plan the iterator ends up with
    # is a policy-conformant one.
    monkeypatch.setattr(isolver, "_get_example_plan_computer", lambda config: (lambda d, p, c: iter([]), {}, "none"))
    monkeypatch.setattr(isolver, "solve_step", lambda **kwargs: (Result.SUCCESS, DummyPolicy(), []))

    def fake_execute_policy(domain, problem, policy, config, time_limit=None, out_actions=None):
        if out_actions is not None:
            out_actions.extend(trajectory)
        return [str(action) for action in trajectory]

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    iterators: list[ProblemIterator] = []

    class CapturingProblemIterator(ProblemIterator):
        def __iter__(self):
            iterators.append(self)
            return super().__iter__()

    monkeypatch.setattr(isolver, "ProblemIterator", CapturingProblemIterator)

    _, _, stats = isolver.solve_iteratively(domain, [problem], solver_config(**config_overrides))
    return iterators[0], stats, problem


def test_a_solving_round_records_its_trajectory_as_an_example_plan(monkeypatch, simple_blocks):
    """(b) After a validation round the iterator's plan set for that problem holds the
    trajectory, and the complexity level the round refuted is invalidated again."""
    domain, _ = simple_blocks
    iterator, stats, problem = run_one_round(monkeypatch, simple_blocks)

    plans = iterator.active_plans[problem.name]
    assert len(plans) == 1
    a, b, c = constants("a b c")
    assert plan_key(plans[0]) == plan_key(Plan([("pick", [a, b]), ("put", [a, c])]))
    assert iterator.policy_plan_counts[problem.name] == 1
    assert stats["policyPlansAdded"] == 1

    # The success refuted complexity 2 (see set_last_result); recording the plan changed the
    # state space, so that refutation is dropped again.
    assert iterator.refuted_complexity == 1

    # The recorded plan is a real, root-anchored plan for this problem.
    assert any(check_formula(state, problem.goal) for state in plan_visited_states(domain, problem, plans[0]))


def test_policy_conformant_plans_off_records_nothing(monkeypatch, simple_blocks):
    """(c) With the option off nothing is added and the round's refutation stands, exactly as
    before this mechanism existed."""
    iterator, stats, problem = run_one_round(monkeypatch, simple_blocks, policy_conformant_plans=False)

    assert iterator.active_plans[problem.name] == []
    assert iterator.policy_plan_counts == {}
    assert "policyPlansAdded" not in stats
    assert iterator.refuted_complexity == 2
    assert iterator.enforce_highest_complexity() is True


def test_policy_conformant_plans_are_inert_without_example_plans(monkeypatch, simple_blocks):
    """The default is true, so it must be a no-op for the rule-based types, whose state space is
    not plan-restricted at all."""
    iterator, stats, problem = run_one_round(monkeypatch, simple_blocks, use_example_plans=False)

    assert iterator.active_plans == {}
    assert "policyPlansAdded" not in stats
