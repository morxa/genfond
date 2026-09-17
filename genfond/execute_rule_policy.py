import logging
import random
import time
from typing import Collection, Mapping, Optional

import dlplan.core
from dlplan.core import InstanceInfo, SyntacticElementFactory
from pddl.action import Action
from pddl.core import Domain, Problem

from genfond.ground import action_string, state_string

from .feature_generator import (
    Feature,
    _get_state_from_goal,
    construct_instance_info,
    construct_vocabulary_info,
    get_action_augmented_state,
)
from .generate_rule_policy import feature_eval_to_cond
from .ground import ground
from .policy import PolicyType
from .rule_policy import Cond, Effect, Policy
from .state_space_generator import State, apply_action_effects, check_formula

log = logging.getLogger("genfond.execution.rule")


class PolicyExecutionError(RuntimeError):

    def __init__(self, message: str):
        super().__init__(message)


class NoActionError(PolicyExecutionError):

    def __init__(self, trace: dict[State, State], state: State):
        self.trace = trace
        self.state = state
        super().__init__(f"No action found for state: {", ".join([str(p) for p in state])}")


class CycleError(PolicyExecutionError):

    def __init__(self, trace: dict[State, State], cycle: list[State]):
        self.trace = trace
        self.cycle = cycle
        super().__init__("Cycle detected")


class ExecutionTimeout(PolicyExecutionError):
    """Raised when a single `execute_*_policy` call runs past its `time_limit`.

    Used by in-loop validation (`validation_time_limit`, see `iterative_solver.
    _test_policy_on_problems`) to bound the wall time a slow/looping candidate policy can spend
    on one problem; the final verification loop in `__main__` never passes a `time_limit`, so
    this is never raised there. A subclass of `PolicyExecutionError` (hence `RuntimeError`), so
    every existing `except RuntimeError:` around `execute_policy` already treats it as an
    ordinary execution failure without needing to name it explicitly.
    """

    def __init__(self, trace: dict[State, State], state: State, time_limit: Optional[float]):
        self.trace = trace
        self.state = state
        self.time_limit = time_limit
        super().__init__(f"Execution exceeded time_limit={time_limit}s")


def _get_dlplan_state(
    instance: InstanceInfo,
    mapping: dict,
    problem: Problem,
    state: State,
    config: dict,
    action: Optional[Action] = None,
) -> dlplan.core.State:
    try:
        return dlplan.core.State(
            -1,
            instance,
            [mapping[predicate] for predicate in get_action_augmented_state(problem, state, config, action)],
        )
    except KeyError as e:
        log.critical(f'Cannot find predicate in mapping {"\n".join(f"{k}: {v}" for k, v in mapping.items())}: {e}')
        raise


def eval_state(
    instance: InstanceInfo,
    mapping: dict,
    features: dict[str, Feature],
    problem: Problem,
    state: State,
    config: dict,
    action: Optional[Action] = None,
) -> dict[str, int]:
    fstate = _get_dlplan_state(instance, mapping, problem, state, config, action)
    feature_eval = dict()
    for fstring, feature in features.items():
        feature_eval[fstring] = feature.evaluate(fstate)
    return feature_eval


def bool_eval_state(
    instance: InstanceInfo,
    mapping: dict,
    features: dict[str, Feature],
    problem: Problem,
    state: State,
    config: dict,
    action: Optional[Action] = None,
) -> dict[str, Cond]:
    feature_eval = eval_state(instance, mapping, features, problem, state, config, action)
    log.debug(f"feature eval: {feature_eval}")
    bool_feature_eval = dict()
    for feature, eval in feature_eval.items():
        # logger=log routes the per-condition "eval to cond" DEBUG line to this module's
        # genfond.execution.rule logger (a child of genfond.execution, silenced to CRITICAL by
        # default -- see the log: map in config/default.yaml) instead of
        # feature_eval_to_cond's own genfond.generation.rule logger, which -v does not silence.
        # This call site runs once per feature per execution step, so left unrouted it was the
        # dominant source of per-step log noise during in-loop validation.
        bool_feature_eval[feature] = feature_eval_to_cond(feature, eval, logger=log)
    return bool_feature_eval


def eval_state_diff(state_eval: dict[str, int], succ_eval: dict[str, int]) -> frozenset[tuple[str, Effect]]:
    log.debug(f"eval state diff: {state_eval} -> {succ_eval}")
    diff = set()
    for feature in succ_eval:
        if state_eval[feature] == succ_eval[feature]:
            continue
        elif feature.startswith("b_"):
            if not state_eval[feature] and succ_eval[feature]:
                diff.add((feature, Effect.SET))
            elif state_eval[feature] and not succ_eval[feature]:
                diff.add((feature, Effect.UNSET))
            else:
                raise RuntimeError(f"Inconsistent state transition: {state_eval} -> {succ_eval}")
        elif feature.startswith("n_"):
            if state_eval[feature] < succ_eval[feature]:
                diff.add((feature, Effect.INCREASE))
            elif state_eval[feature] > succ_eval[feature]:
                diff.add((feature, Effect.DECREASE))
            else:
                raise RuntimeError(f"Inconsistent state transition: {state_eval} -> {succ_eval}")
        else:
            raise ValueError(f"Unknown feature type: {feature}")
    return frozenset(diff)


def state_satisfies_rule_conds(bool_feature_eval: dict[str, Cond], rule_conds: Mapping[str, Cond]) -> bool:
    for feature, cond in rule_conds.items():
        if cond != bool_feature_eval[feature]:
            return False
    return True


def get_next_state(states: Collection[State], _) -> State:
    return random.choice([state for state in states])


def execute_rule_policy(
    domain: Domain,
    problem: Problem,
    policy: Policy,
    config: dict,
    time_limit: Optional[float] = None,
    out_actions: Optional[list[Action]] = None,
) -> list[str]:
    """Run `policy` from `problem.init` and return the action strings it applied.

    `out_actions`, when given, is additionally filled with the *ground actions* themselves, in
    the order they were applied -- the same trajectory the returned strings describe, but in a
    form a `pddl.core.Plan` can be rebuilt from (see `problem_iterator.plan_from_actions`).
    Existing callers pass nothing and are unaffected. The list is appended to as execution
    proceeds, so on a failure it holds the prefix taken before the failure; only a call that
    returns normally has a trajectory that actually reaches the goal.
    """
    log.info(
        f"Executing policy:\n{policy}\nin {domain.name} for problem {problem.name} with features {policy.features}"
    )
    # A monotonic-clock deadline checked once per step, rather than a signal-based interrupt:
    # execution always runs in-process on the caller's own thread (there is no safe way to
    # interrupt it from outside without signals, and this loop already runs inside __main__'s
    # own SIGINT/SIGTERM handler installation, so nesting another signal is best avoided).
    # policy_steps (a step *count*) cannot substitute for this: a single step -- e.g. the
    # sorted(grounded_actions, ...) scan and per-successor evaluation below -- is not itself
    # time-bounded, so a policy stuck re-evaluating a large action set could still run
    # arbitrarily long between step increments.
    deadline = time.monotonic() + time_limit if time_limit else None
    vocabulary = construct_vocabulary_info(domain, config)
    factory = SyntacticElementFactory(vocabulary)
    instance, mapping = construct_instance_info(vocabulary, domain, problem, 0, config)
    features = dict()
    for feature in policy.features:
        if feature.startswith("b_"):
            features[feature] = factory.parse_boolean(feature)
        elif feature.startswith("n_"):
            features[feature] = factory.parse_numerical(feature)
        else:
            raise ValueError(f"Unknown feature type: {feature}")
    # TODO _get_state_from_goal is internal
    goal_state = _get_state_from_goal(problem.goal)
    log.debug("Grounding actions...")
    grounded_actions = ground(domain, problem)
    log.debug("Grounding actions done.")
    state = problem.init
    trace: dict[State, State] = dict()
    num_steps = 0
    actions_taken = []
    max_steps = config["policy_steps"]
    while not check_formula(state, problem.goal) and (max_steps <= 0 or num_steps < max_steps):
        if deadline is not None and time.monotonic() > deadline:
            log.warning(f"Execution exceeded time_limit={time_limit}s, aborting")
            raise ExecutionTimeout(trace, state, time_limit)
        if config["abort_on_cycle"]:
            if state in trace:
                log.error("Cycle detected!")
                cycle = []
                while state not in cycle:
                    cycle.append(state)
                    state = trace[state]
                raise CycleError(trace, cycle)
        log.info(f'New state: {",".join([str(p) for p in state])}')
        feature_eval = eval_state(instance, mapping, features, problem, state, config)
        bool_feature_eval = bool_eval_state(instance, mapping, features, problem, state, config)
        enabled_rules = {rule for rule in policy.rules if state_satisfies_rule_conds(bool_feature_eval, rule.conds)}
        log.debug("Enabled rules: {}".format(",  ".join([str(r) for r in enabled_rules])))
        enabled_constraints = {
            constraint
            for constraint in policy.constraints
            if state_satisfies_rule_conds(bool_feature_eval, constraint.conds)
        }
        log.debug("Enabled constraints: {}".format(",  ".join([str(c) for c in enabled_constraints])))
        if not enabled_rules:
            log.error("No rule enabled!")
            raise RuntimeError("No rule enabled!")
        found_rule = False
        log.debug(
            "Enabled actions: {}".format(
                ", ".join([action_string(a) for a in grounded_actions if check_formula(state, a.precondition)])
            )
        )
        for action in sorted(grounded_actions, key=lambda _: random.random()):
            if not check_formula(state, action.precondition):
                continue
            succs = apply_action_effects(state, action)
            log.debug(
                "Action {} has {} successors: {}".format(
                    action_string(action),
                    len(succs),
                    "; ".join([state_string(s) for s in succs]),
                )
            )
            succs_evals = [eval_state(instance, mapping, features, problem, succ, config) for succ in succs]
            log.debug(f"succs_evals: {succs_evals}")
            succs_diffs = {eval_state_diff(feature_eval, succ_eval) for succ_eval in succs_evals}
            log.debug(f'succs_diffs:\n{"\n".join([", ".join([str(d) for d in ds]) for ds in succs_diffs])}')
            ok = True
            for constraint in enabled_constraints:
                if constraint.effs & succs_diffs:
                    log.info(f"Constraint {constraint} violated!")
                    ok = False
                    break
            bool_succs_evals = [bool_eval_state(instance, mapping, features, problem, succ, config) for succ in succs]
            for bool_succs_eval in bool_succs_evals:
                for state_constraint in policy.state_constraints:
                    violated = True
                    for feature, cond in state_constraint.conds.items():
                        if cond != bool_succs_eval[feature]:
                            violated = False
                            break
                    if violated:
                        log.info(f"State constraint {state_constraint} violated!")
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for rule in enabled_rules:
                log.debug(f"Checking rule {rule}")
                if rule.effs == succs_diffs or policy.type == PolicyType.CONSTRAINED and rule.effs & succs_diffs:
                    found_rule = True
                    log.info(f"Found matching rule:\n{rule}")
                    log.info(f"Applying action {action_string(action)}")
                    new_state = get_next_state(succs, action)
                    trace[state] = new_state
                    state = new_state
                    num_steps += 1
                    actions_taken.append(action_string(action))
                    if out_actions is not None:
                        out_actions.append(action)
                    break
            if found_rule:
                break
        if not found_rule:
            log.error(f"No matching rule found for diff {succs_diffs}!")
            log.error("Enabled rules:\n  {}".format("\n  ".join([str(r) for r in enabled_rules])))
            raise RuntimeError("No matching rule found!")
    if not check_formula(state, problem.goal):
        log.error("Goal not reached!")
        raise RuntimeError("Goal not reached!")
    log.info("Goal reached!")
    return actions_taken
    return actions_taken
