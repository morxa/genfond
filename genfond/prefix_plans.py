"""Policy-prefix example plans (H30).

When the loop adds a problem ``Q`` to the training set it adds it *because the best policy so
far, ``P``, fails on it*. ``Q``'s example plans, however, come from SIW and have nothing to do
with what ``P`` does: on the enlarged, plan-restricted instance there is then no model anywhere
near ``P``'s cost, and the round drifts into a patchwork policy fitted to the sampled plans.
``policy_conformant_plans`` (H27) cannot help here -- it records ``P``'s trajectories only on the
problems ``P`` *solves*, and ``Q`` is by definition not one of them.

This module builds the missing plan: run ``P`` on ``Q``, keep the prefix it got right, and let
the planner finish the job from there.

1. Execute ``P`` on ``Q`` with ``out_actions`` (the H27 mechanism) and cut the trajectory at the
   point where ``P`` went wrong -- at the first visit of the state it later loops back to when it
   cycles, at the end of the trajectory when it runs out of applicable actions, and at
   ``prefix_plan_max_length`` in any case.
2. Re-root ``Q`` at the state that prefix reaches and ask the planner for up to
   ``prefix_plan_count`` plans from there (exactly what ``frontier.expand_frontier`` does for a
   frontier state), then splice prefix and suffix into one root-anchored plan --
   ``StateSpaceGraph`` always replays a plan from ``problem.init``.
3. If the full prefix yields no plan, back off: drop the last 1, 2, 4, ... actions
   (``prefix_plan_backoff`` attempts). The steps a policy takes just before it loops are the
   ones most likely to have been the wrong ones.

The planner is incomplete and the re-rooted state may genuinely be a dead end, so "no plan" is
not evidence of anything; it simply means ``Q`` keeps only its SIW plans, exactly as before this
mechanism existed.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from pddl.action import Action
from pddl.core import Domain, Plan, Problem

from .execute_policy import execute_policy
from .execute_rule_policy import CycleError, ExecutionTimeout, NoActionError
from .frontier import PlannerComputePlans, _root_anchored_plan, reroot
from .problem_iterator import plan_from_actions
from .state_space_generator import State, apply_action_effects, check_formula

log = logging.getLogger("genfond.prefix_plans")


@dataclass(frozen=True)
class PrefixPlans:
    """What `policy_prefix_plans` found for one newly added problem.

    `attempted` separates "the mechanism ran and found nothing" (a failure worth counting) from
    "there was nothing to run" -- no best policy yet, or the mechanism switched off -- which is
    the normal state of affairs for the first problem of a run.
    """

    plans: tuple[Plan, ...] = ()
    prefix_length: int = 0
    backoff: int = 0
    attempted: bool = False


def executed_states(
    problem: Problem,
    actions: Sequence[Action],
    trace: Optional[Mapping[State, State]] = None,
) -> Optional[list[State]]:
    """The states `actions` visits from `problem.init`, as `[s0, ..., sn]` with `n = len(actions)`.

    Every step is verified against the action it belongs to, so the result is always a real path
    through the problem's state space. A deterministic action fixes its own successor; for a
    nondeterministic one the executor's `trace` (which records the outcome that was actually
    sampled) decides, and if that is unavailable or does not name a successor of this state
    there is no way to know where the execution went -- `None` is returned and the caller gives
    up on this problem rather than guess.
    """
    states = [problem.init]
    state = problem.init
    for action in actions:
        if not check_formula(state, action.precondition):
            return None
        successors = apply_action_effects(state, action)
        if len(successors) == 1:
            state = next(iter(successors))
        elif trace is not None and trace.get(state) in successors:
            state = trace[state]
        else:
            return None
        states.append(state)
    return states


def prefix_cut(states: Sequence[State], max_length: int) -> int:
    """How many of the executed actions to keep: the index of the prefix's last state.

    A repeated state means the policy looped, and everything from the *first* visit of that
    state onwards is the loop -- the prefix ends there. Otherwise the whole trajectory is kept
    (the policy ran out of applicable actions at its end, or its execution was cut short). Either
    way the prefix is capped at `max_length`: nothing else bounds the length of a trajectory that
    ran until `policy_steps` or until the validation time limit.
    """
    first_visit: dict[State, int] = dict()
    for index, state in enumerate(states):
        if state in first_visit:
            return min(first_visit[state], max_length)
        first_visit[state] = index
    return min(len(states) - 1, max_length)


def _plans_from(
    domain: Domain,
    problem: Problem,
    prefix: Sequence[Action],
    state: State,
    planner: PlannerComputePlans,
    planner_config: Mapping[str, Any],
    count: int,
) -> list[Plan]:
    """Ask the planner for up to `count` plans from `state` and splice them onto `prefix`."""
    rerooted = reroot(problem, state, f"prefix-{len(prefix)}")
    log.debug(
        "Planning from the policy prefix of %s (%d steps from the initial state)",
        problem.name,
        len(prefix),
    )
    # The planner yields lazily, so islice only ever pays for the plans that are consumed.
    suffixes = list(itertools.islice(planner(str(domain), str(rerooted), dict(planner_config)), count))
    return [_root_anchored_plan(prefix, suffix) for suffix in suffixes]


def policy_prefix_plans(
    domain: Domain,
    problem: Problem,
    policy: Any,
    config: Mapping[str, Any],
    planner: PlannerComputePlans,
    planner_config: Mapping[str, Any],
) -> PrefixPlans:
    """Build root-anchored example plans for `problem` out of what `policy` already gets right.

    `policy` is the best policy so far; `problem` is the one just added to the training set,
    which that policy fails on. Returns the plans to prepend to `problem`'s example plans, or an
    attempted-but-empty result when the policy cycles at the initial state, the execution cannot
    be replayed, or the planner finds nothing from any of the prefixes tried.
    """
    max_length = config.get("prefix_plan_max_length") or 200
    count = config.get("prefix_plan_count") or 1
    backoff_attempts = config.get("prefix_plan_backoff") or 0

    actions: list[Action] = []
    trace: Optional[Mapping[State, State]] = None
    try:
        execute_policy(
            domain,
            problem,
            policy,
            config,
            time_limit=config.get("validation_time_limit"),
            out_actions=actions,
        )
    except (CycleError, NoActionError, ExecutionTimeout) as e:
        # All three carry the trace of the execution; the exception type only says *why* the
        # policy stopped, and `prefix_cut` reads that off the state sequence itself.
        trace = e.trace
    except RuntimeError:
        # "Goal not reached" after `policy_steps` (a `PolicyExecutionError`, hence a
        # `RuntimeError`), or anything else the executor raises: the
        # actions applied so far are still a real trajectory, there is just no trace to
        # disambiguate a nondeterministic outcome with.
        pass
    else:
        # The policy solved the problem after all -- execution is randomized, so a policy that
        # failed during validation can get through here. Its own trajectory is then the best
        # example plan there is and needs no planner at all.
        if actions:
            log.info("The best policy solves %s on this execution; recording its trajectory", problem.name)
            return PrefixPlans(plans=(plan_from_actions(actions),), prefix_length=len(actions), attempted=True)
        return PrefixPlans(attempted=True)

    if not actions:
        log.info("The best policy applies no action at all on %s; no policy-prefix plan", problem.name)
        return PrefixPlans(attempted=True)
    states = executed_states(problem, actions, trace)
    if states is None:
        log.warning("Cannot replay the best policy's execution on %s; no policy-prefix plan", problem.name)
        return PrefixPlans(attempted=True)
    cut = prefix_cut(states, max_length)
    if cut <= 0:
        # The policy loops straight back to the initial state, so the prefix is empty and
        # planning from it is just the planner's own job on the unmodified problem, which the
        # `min_number_of_plans` batch already does.
        log.info("The best policy returns to the initial state of %s; no policy-prefix plan", problem.name)
        return PrefixPlans(attempted=True)

    # Backoff 0 is the full prefix; then drop the last 1, 2, 4, ... actions.
    for backoff, drop in enumerate([0] + [2**attempt for attempt in range(backoff_attempts)]):
        length = cut - drop
        if length <= 0:
            break
        plans = _plans_from(domain, problem, actions[:length], states[length], planner, planner_config, count)
        if plans:
            return PrefixPlans(plans=tuple(plans), prefix_length=length, backoff=backoff, attempted=True)
    log.info(
        "No plan from any policy prefix of %s (prefix length %d, %d backoff attempt(s))",
        problem.name,
        cut,
        backoff_attempts,
    )
    return PrefixPlans(attempted=True)
