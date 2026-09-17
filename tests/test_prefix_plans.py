"""Tests for policy-prefix example plans (H30).

A problem joins the training set precisely because the best policy so far fails on it, and the
SIW plans it is seeded with have nothing to do with what that policy does. `genfond.prefix_plans`
closes that gap: it runs the policy on the new problem, cuts the trajectory where the policy went
wrong, and lets the planner finish from there.

The prefix-building tests run the *real* datalog executor over a tiny in-memory "corridor" domain
whose only cycle-free route to the goal needs an action the policy does not have, so the policy
provably loops; only the planner is mocked. The iterator- and solver-level tests use stubs, as
`tests/test_policy_plans.py` does.
"""

from pddl.action import Action
from pddl.core import Domain, Plan, Problem
from pddl.logic import Predicate, constants, variables
from pddl.requirements import Requirements

from genfond import iterative_solver as isolver
from genfond.config_handler import ConfigHandler
from genfond.datalog_policy import DatalogPolicy, DatalogPolicyRule
from genfond.ground import ground
from genfond.prefix_plans import PrefixPlans, executed_states, policy_prefix_plans, prefix_cut
from genfond.problem_iterator import ProblemIterator, Result, plan_key
from genfond.state_space_generator import StateSpaceGraph, check_formula, plan_visited_states

# --- the corridor domain ----------------------------------------------------------------------

L0, L1, L2, L3, L4 = constants("l0 l1 l2 l3 l4")
_x, _y = variables("x y")
AT = Predicate("at", _x)
EDGE = Predicate("edge", _x, _y)
LINK = Predicate("link", _x, _y)


def corridor_domain() -> Domain:
    """`go` follows `edge`, `jump` follows `link`; both just move the token."""
    return Domain(
        "corridor",
        requirements=[Requirements.STRIPS],
        predicates=[AT, EDGE, LINK],
        actions=[
            Action("go", parameters=[_x, _y], precondition=AT(_x) & EDGE(_x, _y), effect=AT(_y) & ~AT(_x)),
            Action("jump", parameters=[_x, _y], precondition=AT(_x) & LINK(_x, _y), effect=AT(_y) & ~AT(_x)),
        ],
    )


def corridor_problem(domain: Domain, name: str = "p1") -> Problem:
    """l0 -> l1 -> l2 <-> l3, and the goal l4 is only reachable from l3 by `jump`.

    Every state has exactly one outgoing `edge`, so a policy that only knows `go` walks
    l0, l1, l2, l3 and is then thrown back to l2 -- a cycle whose first repeated state is l2.
    """
    return Problem(
        name,
        domain=domain,
        requirements=[Requirements.STRIPS],
        objects=[L0, L1, L2, L3, L4],
        init=[AT(L0), EDGE(L0, L1), EDGE(L1, L2), EDGE(L2, L3), EDGE(L3, L2), LINK(L3, L4)],
        goal=AT(L4),
    )


def go_only_policy() -> DatalogPolicy:
    """Takes the single applicable `go` at every state, and therefore never reaches l4."""
    return DatalogPolicy([DatalogPolicyRule("go(X, Y)")])


def datalog_config(**overrides):
    return ConfigHandler(type="datalog", override={"abort_on_cycle": True, **overrides})


class RecordingPlanner:
    """A `PlannerComputePlans` stub that returns a canned answer per call and records the calls."""

    def __init__(self, *answers: list[Plan]):
        self.answers = list(answers)
        self.calls: list[str] = []

    def __call__(self, domain_str, problem_str, planner_config):
        self.calls.append(problem_str)
        answer = self.answers.pop(0) if self.answers else []
        return iter(answer)


def suffix_from_l2() -> Plan:
    return Plan([("go", [L2, L3]), ("jump", [L3, L4])])


# --- (a) the prefix is cut at the first repeated state -----------------------------------------


def test_prefix_is_cut_at_the_first_repeated_state():
    domain, problem = corridor_domain(), corridor_problem(corridor_domain())
    planner = RecordingPlanner([suffix_from_l2()])

    result = policy_prefix_plans(domain, problem, go_only_policy(), datalog_config(), planner, {})

    # The policy walks l0, l1, l2, l3 and is thrown back to l2: the first repeated state is l2,
    # first visited after two actions, so the prefix is exactly those two.
    assert result.attempted is True
    assert result.prefix_length == 2
    assert result.backoff == 0
    assert plan_key(result.plans[0]) == plan_key(
        Plan([("go", [L0, L1]), ("go", [L1, L2]), ("go", [L2, L3]), ("jump", [L3, L4])])
    )
    # The planner was asked to plan from the end of that prefix, not from the initial state.
    assert len(planner.calls) == 1
    assert "(at l2)" in planner.calls[0] and "(at l0)" not in planner.calls[0]


def test_the_prefix_is_capped_at_prefix_plan_max_length():
    domain, problem = corridor_domain(), corridor_problem(corridor_domain())
    planner = RecordingPlanner([Plan([("go", [L1, L2]), ("go", [L2, L3]), ("jump", [L3, L4])])])

    result = policy_prefix_plans(
        domain, problem, go_only_policy(), datalog_config(prefix_plan_max_length=1), planner, {}
    )

    assert result.prefix_length == 1
    assert "(at l1)" in planner.calls[0]


def test_prefix_cut_takes_the_whole_trajectory_when_nothing_repeats():
    """`prefix_cut` is the only place the cut rule lives, so pin both of its branches down."""
    states = [frozenset({AT(L0)}), frozenset({AT(L1)}), frozenset({AT(L2)})]
    assert prefix_cut(states, 200) == 2
    assert prefix_cut(states, 1) == 1
    # A repeat cuts at the *first* visit of the repeated state, not at the repetition.
    assert prefix_cut(states + [frozenset({AT(L1)})], 200) == 1


def test_executed_states_verifies_every_step_against_its_action():
    domain, problem = corridor_domain(), corridor_problem(corridor_domain())
    go = {(a.name, a.parameters): a for a in ground(domain, problem)}
    first, second = go[("go", (L0, L1))], go[("go", (L1, L2))]

    states = executed_states(problem, [first, second])
    assert states is not None
    assert len(states) == 3 and states[0] == problem.init
    assert AT(L2) in states[2]

    # An action whose precondition does not hold cannot have been executed here.
    assert executed_states(problem, [second, first]) is None


# --- (b) the spliced plan is root-anchored -----------------------------------------------------


def test_the_spliced_plan_is_root_anchored_and_replays_to_a_goal():
    domain, problem = corridor_domain(), corridor_problem(corridor_domain())
    planner = RecordingPlanner([suffix_from_l2()])

    result = policy_prefix_plans(domain, problem, go_only_policy(), datalog_config(), planner, {})
    plan = result.plans[0]

    # Replayed from problem.init -- which is where StateSpaceGraph replays a plan from -- it
    # reaches a goal state.
    visited = plan_visited_states(domain, problem, plan)
    assert problem.init in visited
    assert any(check_formula(state, problem.goal) for state in visited)

    graph = StateSpaceGraph(domain, problem, plans=[plan])
    assert any(node.goal for node in graph.nodes.values())


# --- (c) backoff -------------------------------------------------------------------------------


def test_backoff_drops_the_last_actions_when_the_full_prefix_has_no_plan():
    domain, problem = corridor_domain(), corridor_problem(corridor_domain())
    # Nothing from l2 (the full prefix), a plan from l1 (one action dropped).
    planner = RecordingPlanner([], [Plan([("go", [L1, L2]), ("go", [L2, L3]), ("jump", [L3, L4])])])

    result = policy_prefix_plans(domain, problem, go_only_policy(), datalog_config(), planner, {})

    assert (result.prefix_length, result.backoff) == (1, 1)
    assert len(planner.calls) == 2
    assert "(at l2)" in planner.calls[0] and "(at l1)" in planner.calls[1]
    assert plan_key(result.plans[0]) == plan_key(
        Plan([("go", [L0, L1]), ("go", [L1, L2]), ("go", [L2, L3]), ("jump", [L3, L4])])
    )


def test_no_plan_from_any_prefix_is_an_attempted_failure():
    domain, problem = corridor_domain(), corridor_problem(corridor_domain())
    planner = RecordingPlanner()  # never returns anything

    result = policy_prefix_plans(domain, problem, go_only_policy(), datalog_config(), planner, {})

    assert result == PrefixPlans(attempted=True)
    # The full prefix (l2) plus the one backoff step that does not empty it (l1); dropping 2 or 4
    # of 2 actions leaves nothing to plan from, so the remaining attempts are skipped.
    assert len(planner.calls) == 2


def test_a_policy_that_loops_back_to_the_initial_state_yields_nothing():
    """The prefix is then empty, and planning from it is just the planner's own job on the
    unmodified problem -- which the min_number_of_plans batch already does."""
    domain = corridor_domain()
    problem = Problem(
        "loop",
        domain=domain,
        requirements=[Requirements.STRIPS],
        objects=[L0, L1, L2],
        init=[AT(L0), EDGE(L0, L1), EDGE(L1, L0), LINK(L1, L2)],
        goal=AT(L2),
    )
    planner = RecordingPlanner([Plan([("jump", [L1, L2])])])

    result = policy_prefix_plans(domain, problem, go_only_policy(), datalog_config(), planner, {})

    assert result == PrefixPlans(attempted=True)
    assert planner.calls == []


def test_a_policy_that_solves_the_problem_records_its_own_trajectory():
    """Execution is randomized, so a policy that failed validation can get through here; its own
    trajectory is then the best example plan there is and needs no planner at all."""
    domain = corridor_domain()
    problem = Problem(
        "solvable",
        domain=domain,
        requirements=[Requirements.STRIPS],
        objects=[L0, L1],
        init=[AT(L0), EDGE(L0, L1)],
        goal=AT(L1),
    )
    planner = RecordingPlanner([Plan([("go", [L0, L1])])])

    result = policy_prefix_plans(domain, problem, go_only_policy(), datalog_config(), planner, {})

    assert plan_key(result.plans[0]) == plan_key(Plan([("go", [L0, L1])]))
    assert result.attempted is True
    assert planner.calls == []


def test_a_trajectory_without_a_trace_is_still_cut_at_the_repeat():
    """With abort_on_cycle off the executor runs to `policy_steps` and raises a plain
    PolicyExecutionError, which carries no trace -- the state sequence is replayed instead."""
    domain, problem = corridor_domain(), corridor_problem(corridor_domain())
    planner = RecordingPlanner([suffix_from_l2()])
    config = ConfigHandler(type="datalog", override={"abort_on_cycle": False, "policy_steps": 12})

    result = policy_prefix_plans(domain, problem, go_only_policy(), config, planner, {})

    assert result.prefix_length == 2
    assert "(at l2)" in planner.calls[0]


# --- ProblemIterator-level tests ---------------------------------------------------------------


class DummyProblem:
    def __init__(self, name, objects=()):
        self.name = name
        self.objects = objects


def a_plan(*names):
    return Plan([(name, [L0]) for name in names])


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


def a_started_iterator(config, hook, num_plans=10):
    problems = [DummyProblem("p1"), DummyProblem("p2")]
    plans = {p.name: iter([a_plan(f"{p.name}-{i}") for i in range(num_plans)]) for p in problems}
    iterator = iter(ProblemIterator(problems, config, plans=plans, on_problem_added=hook))
    return iterator, next(iterator)


def test_prefix_plans_go_in_front_and_are_exempt_from_the_cap_and_the_floor():
    calls: list[str] = []

    def hook(problem):
        calls.append(problem.name)
        return PrefixPlans(plans=(a_plan("prefix-step"),), prefix_length=3, backoff=1, attempted=True)

    iterator, _ = a_started_iterator(iterator_config(min_number_of_plans=2, max_plans_per_problem=2), hook)

    assert calls == ["p1"]
    plans = iterator.active_plans["p1"]
    # The prefix plan is in front of the floor batch, which was drawn in full on top of it.
    assert plan_key(plans[0]) == plan_key(a_plan("prefix-step"))
    assert len(plans) == 3
    assert iterator.policy_plan_counts["p1"] == 1
    assert iterator.prefix_plans_added == 1
    # Two *capped* plans, so the cap is reached -- the prefix plan does not count towards it.
    assert iterator._plan_cap_reached("p1") is True


def test_prefix_plans_invalidate_the_refuted_complexity_level():
    """`_add_next_problem` normally carries `succ_complexity - 1` over, because adding an
    instance is monotone. Seeding that instance with extra plans is a different state space than
    the one the refutation was established over, so the bound is dropped."""
    hook = lambda problem: PrefixPlans(plans=(a_plan("prefix-step"),), attempted=True)
    iterator, _ = a_started_iterator(iterator_config(), hook)
    iterator.set_last_result(Result.NO_SOLUTION)
    next(iterator)  # complexity 3
    iterator.set_last_result(Result.SUCCESS, cost=(4,))
    iterator.set_solved(iterator.problems[0], True)

    next(iterator)  # the add-a-problem branch, which seeds p2

    assert [p.name for p in iterator.active_problems] == ["p1", "p2"]
    assert iterator.refuted_complexity == iterator.config["min_complexity"] - 1
    assert iterator.enforce_highest_complexity() is False


def test_an_empty_result_counts_as_a_failure_only_when_the_hook_actually_ran():
    attempted = [PrefixPlans(attempted=True), PrefixPlans()]
    iterator, _ = a_started_iterator(iterator_config(), lambda problem: attempted.pop(0))
    assert (iterator.prefix_plans_added, iterator.prefix_plan_failures) == (0, 1)

    # cost 3 -> max_cost 2, which stops the ladder climbing complexity and adds p2 instead.
    iterator.set_last_result(Result.SUCCESS, cost=(3,))
    iterator.set_solved(iterator.problems[0], True)
    next(iterator)  # p2 joins; the hook returns a not-attempted result this time

    assert (iterator.prefix_plans_added, iterator.prefix_plan_failures) == (0, 1)
    assert iterator.active_plans["p2"] == [a_plan("p2-0")]


def test_a_prefix_plan_the_problem_already_has_is_not_added_twice():
    """The floor batch is drawn first, so a prefix plan identical to one of its plans would only
    duplicate work in every later round's state space."""
    hook = lambda problem: PrefixPlans(plans=(a_plan("p1-0"),), attempted=True)
    iterator, _ = a_started_iterator(iterator_config(), hook)

    assert [plan_key(plan) for plan in iterator.active_plans["p1"]] == [plan_key(a_plan("p1-0"))]
    assert iterator.prefix_plans_added == 0
    assert iterator.prefix_plan_failures == 1


def test_without_a_hook_nothing_is_seeded():
    """(d) The option off means `on_problem_added=None`, which is what every other caller and
    test passes: byte-for-byte the old behaviour."""
    iterator, _ = a_started_iterator(iterator_config(), None)

    assert [plan_key(plan) for plan in iterator.active_plans["p1"]] == [plan_key(a_plan("p1-0"))]
    assert iterator.policy_plan_counts == {}
    assert (iterator.prefix_plans_added, iterator.prefix_plan_failures) == (0, 0)


# --- solve_iteratively wiring ------------------------------------------------------------------


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
        "min_number_of_plans": 1,
        "max_plans_per_problem": None,
        "max_frontier_expansions": 20,
        "use_example_plans": True,
        "frontier_expansion": False,
        "policy_conformant_plans": False,
        "policy_prefix_plans": True,
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


def run_two_problems(monkeypatch, **config_overrides):
    """One successful round on p1 (which the policy solves) followed by p2 joining the training
    set (which it does not). Returns the live iterator, the stats, and the recorded hook calls."""
    domain = corridor_domain()
    p1 = corridor_problem(domain, "p1")
    p2 = Problem(
        "p2",
        domain=domain,
        requirements=[Requirements.STRIPS],
        objects=[L0, L1, L2, L3, L4],
        init=[AT(L0), EDGE(L0, L1), EDGE(L1, L2), EDGE(L2, L3), EDGE(L3, L2), LINK(L3, L4)],
        goal=AT(L3),
    )
    # p1 must sort first, so give p2 one more object.
    p2 = Problem(
        "p2",
        domain=domain,
        requirements=[Requirements.STRIPS],
        objects=[L0, L1, L2, L3, L4],
        init=p2.init,
        goal=p2.goal,
    )
    monkeypatch.setattr(isolver, "_get_example_plan_computer", lambda config: (lambda d, p, c: iter([]), {}, "none"))
    # One policy object for the whole run, so the hook's argument can be compared by identity
    # with the policy solve_iteratively returns.
    the_policy = DummyPolicy()
    monkeypatch.setattr(isolver, "solve_step", lambda **kwargs: (Result.SUCCESS, the_policy, []))

    def fake_execute_policy(domain, problem, policy, config, time_limit=None, out_actions=None):
        if problem.name == "p2":
            raise RuntimeError("policy does not solve p2")
        return ["go(l0,l1)"]

    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    calls: list[tuple] = []

    def fake_prefix_plans(domain, problem, policy, config, planner, planner_config):
        calls.append((problem.name, policy))
        return PrefixPlans(plans=(Plan([("go", [L0, L1])]),), prefix_length=1, backoff=0, attempted=True)

    monkeypatch.setattr(isolver, "policy_prefix_plans", fake_prefix_plans)

    iterators: list[ProblemIterator] = []

    class CapturingProblemIterator(ProblemIterator):
        def __iter__(self):
            iterators.append(self)
            return super().__iter__()

    monkeypatch.setattr(isolver, "ProblemIterator", CapturingProblemIterator)

    policy, _, stats = isolver.solve_iteratively(domain, [p1, p2], solver_config(**config_overrides))
    return iterators[0], stats, calls, policy


def test_a_newly_added_problem_is_seeded_with_policy_prefix_plans(monkeypatch):
    iterator, stats, calls, policy = run_two_problems(monkeypatch)

    # p1 is added before any policy exists, so the hook short-circuits and never reaches
    # policy_prefix_plans; p2 is added after the first success and is seeded from that policy.
    assert [name for name, _ in calls] == ["p2"]
    assert calls[0][1] is policy
    assert plan_key(iterator.active_plans["p2"][0]) == plan_key(Plan([("go", [L0, L1])]))
    assert stats["prefixPlansAdded"] == 1
    assert stats["prefixPlanFailures"] == 0


def test_policy_prefix_plans_off_seeds_nothing(monkeypatch):
    """(d) end to end: no hook is installed, nothing is built, and the stats stay clean."""
    iterator, stats, calls, _ = run_two_problems(monkeypatch, policy_prefix_plans=False)

    assert calls == []
    assert iterator.on_problem_added is None
    assert iterator.active_plans["p2"] == []
    assert "prefixPlansAdded" not in stats


def test_policy_prefix_plans_are_inert_without_example_plans(monkeypatch):
    """The default is true, so it must be a no-op for the rule-based types, whose state space is
    not plan-restricted at all."""
    iterator, stats, calls, _ = run_two_problems(monkeypatch, use_example_plans=False)

    assert calls == []
    assert iterator.on_problem_added is None
    assert "prefixPlansAdded" not in stats
