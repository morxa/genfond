"""Frontier expansion for the datalog policy encoding.

When ``frontier_expansion`` is on, ``StateSpaceGraph`` leaves off-plan successors unexpanded
and marks them ``Alive.PRUNED``. ``solve_datalog.lp`` may select a transition into such a
state on the optimistic assumption that it is solvable, charged at a higher optimization
priority than feature complexity, and reports it as ``frontier/2``.

This module closes the loop: it turns those atoms back into states, re-roots the planning
problem at each of them, and asks the planner for a plan. A plan becomes a new example plan
(prefixed with the path from the root, so it stays anchored at the problem's initial state);
no plan marks the state as a dead end so the solver stops selecting it.

Note that the planner is incomplete, so "no plan" does not prove unsolvability. A false dead
end can only prevent a policy from being found, never produce an incorrect one -- the final
policy is always re-validated by ``execute_policy`` on every problem.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from pddl.action import Action
from pddl.core import Domain, Plan, Problem

from .feature_generator import FeaturePool
from .ground import action_string, state_string
from .state_space_generator import State

log = logging.getLogger("genfond.frontier")

PlannerComputePlans = Callable[[str, str, dict[str, Any]], Iterator[Plan]]


@dataclass(frozen=True)
class FrontierState:
    """An unexpanded state that the last model relied on reaching."""

    problem_name: str
    node_id: int
    state: State
    prefix: tuple[Action, ...]
    """Actions leading from the problem's initial state to `state`."""


def collect_frontier_states(feature_pool: FeaturePool, solution: Mapping[str, Any]) -> list[FrontierState]:
    """Resolve the `frontier/2` atoms of a model into states plus their path from the root."""
    frontier_states = []
    for problem_id, node_id in sorted(solution.get("frontier", set())):
        problem, node = feature_pool.lookup_node(problem_id, node_id)
        prefix = feature_pool.state_graphs[problem.name].action_path_from_root(node)
        if prefix is None:
            log.warning(
                "Frontier state %s of %s is unreachable from the root, skipping it",
                node_id,
                problem.name,
            )
            continue
        frontier_states.append(
            FrontierState(
                problem_name=problem.name,
                node_id=node_id,
                state=node.state,
                prefix=tuple(prefix),
            )
        )
    return frontier_states


def reroot(problem: Problem, state: State, suffix: str) -> Problem:
    """Build a copy of `problem` whose initial state is `state`.

    Passing ``domain=`` instead of ``domain_name=`` makes pddl re-derive the requirements and
    raise when they differ from the problem's own, so the name is used here.
    """
    return Problem(
        f"{problem.name}-frontier-{suffix}",
        domain_name=problem.domain_name,
        requirements=problem.requirements,
        objects=problem.objects,
        init=state,
        goal=problem.goal,
    )


def _root_anchored_plan(prefix: Sequence[Action], suffix: Plan) -> Plan:
    """Splice a plan found from a frontier state onto the path that reaches that state."""
    return Plan([(action.name, list(action.parameters)) for action in prefix] + list(suffix.actions))


def expand_frontier(
    domain: Domain,
    problems: Mapping[str, Problem],
    frontier_states: Sequence[FrontierState],
    planner: PlannerComputePlans,
    planner_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[Plan]], dict[str, set[State]]]:
    """Plan from each frontier state.

    Returns the new example plans per problem name, and the states the planner could not
    solve, which the caller must feed back so the solver stops selecting them.
    """
    limit = config["max_frontier_states_per_round"]
    if limit:
        if len(frontier_states) > limit:
            log.info(
                "Expanding %d of %d selected frontier states this round (max_frontier_states_per_round)",
                limit,
                len(frontier_states),
            )
        frontier_states = frontier_states[:limit]
    new_plans: dict[str, list[Plan]] = dict()
    dead_states: dict[str, set[State]] = dict()
    for frontier_state in frontier_states:
        problem = problems[frontier_state.problem_name]
        rerooted = reroot(problem, frontier_state.state, str(frontier_state.node_id))
        log.info(
            "Planning from frontier state %d of %s (%d steps from the initial state)",
            frontier_state.node_id,
            problem.name,
            len(frontier_state.prefix),
        )
        plan: Optional[Plan] = next(planner(str(domain), str(rerooted), dict(planner_config)), None)
        if plan is None:
            log.warning(
                "No plan from frontier state %d of %s, marking it as a dead end: %s",
                frontier_state.node_id,
                problem.name,
                state_string(frontier_state.state),
            )
            dead_states.setdefault(problem.name, set()).add(frontier_state.state)
            continue
        anchored = _root_anchored_plan(frontier_state.prefix, plan)
        log.info(
            "Found a plan of length %d from frontier state %d of %s, new example plan: %s",
            len(plan.actions),
            frontier_state.node_id,
            problem.name,
            " ".join(action_string(action) for action in anchored.instantiate(domain)),
        )
        new_plans.setdefault(problem.name, []).append(anchored)
    return new_plans, dead_states
