import logging
import random
from typing import Any, Collection, Mapping, Optional

from pddl.action import Action
from pddl.core import Domain, Problem

from wlplan.feature_generation import Features as FeatureGenerator
from wlplan.feature_generation import load_feature_generator
from wlplan.planning import Domain as WlDomain
from wlplan.planning import Problem as WlProblem
from wlplan.planning import to_wlplan_domain, to_wlplan_problem

from .feature_generator_wlplan import WlFeature, get_sub_index_map, to_wlplan_state
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


def eval_state(
    fg: FeatureGenerator,
    features: dict[str, WlFeature],
    wlplan_domain: WlDomain,
    wlplan_problem: WlProblem,
    state: State,
) -> dict[str, int]:
    wlplan_state = to_wlplan_state(state, wlplan_domain, wlplan_problem)
    fg.set_problem(wlplan_problem)
    x = fg.embed(wlplan_state)
    feature_eval = dict()
    for fstring, feature in features.items():
        feature_eval[fstring] = x[feature.id]
    return feature_eval


def bool_eval_state(
    fg: FeatureGenerator,
    features: dict[str, WlFeature],
    wlplan_domain: WlDomain,
    wlplan_problem: WlProblem,
    state: State,
) -> dict[str, Cond]:
    feature_eval = eval_state(fg, features, wlplan_domain, wlplan_problem, state)
    log.debug(f"feature eval: {feature_eval}")
    bool_feature_eval = dict()
    for feature, eval in feature_eval.items():
        bool_feature_eval[feature] = feature_eval_to_cond(feature, eval)
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


def action_string(action: Action) -> str:
    return f'{action.name}({",".join([str(p) for p in action.parameters])})'


def state_string(state: State) -> str:
    return ",".join([str(p) for p in state])


def execute_rule_policy_wl(domain: Domain, problem: Problem, policy: Policy, config: dict) -> list[str]:
    log.info(
        f"Executing policy:\n{policy}\nin {domain.name} for problem {problem.name} with features {policy.features}"
    )
    fg = load_feature_generator(policy.save_file, quiet=True)
    colour_to_layer = fg.get_colour_to_layer()
    wlplan_domain = to_wlplan_domain(pddl_domain=domain)
    wlplan_problem = to_wlplan_problem(pddl_domain=domain, pddl_problem=problem)

    # reverse partition features
    features = dict()
    n_cat_features = fg.get_n_features()
    n_con_features = fg.get_n_features()
    for feature_str in policy.features:
        toks = feature_str.split("_")
        assert len(toks) == 3
        assert toks[0] == "n"
        assert toks[-1] in {"con", "cat", "sub"}
        if toks[-1] == "sub":
            i, j = toks[1].split("-")
            i, j = int(i), int(j)
            colour = max(colour_to_layer[i], colour_to_layer[j])
            id = get_sub_index_map(n_cat_features, n_con_features)[(i, j)]
        else:
            i = int(toks[1])
            colour = colour_to_layer[i]
            id = n_cat_features + i if toks[-1] else i
        features[feature_str] = WlFeature(id=id, name=feature_str, complexity=colour)

    log.debug("Grounding actions...")
    grounded_actions = ground(domain, problem)
    log.debug("Grounding actions done.")

    state = problem.init

    trace: dict[State, State] = dict()
    num_steps = 0
    actions_taken = []
    max_steps = config["policy_steps"]
    while not check_formula(state, problem.goal) and (max_steps <= 0 or num_steps < max_steps):
        if config["abort_on_cycle"]:
            if state in trace:
                log.error("Cycle detected!")
                cycle = []
                while state not in cycle:
                    cycle.append(state)
                    state = trace[state]
                raise CycleError(trace, cycle)
        log.info(f'New state: {",".join([str(p) for p in state])}')
        feature_eval = eval_state(fg, features, wlplan_domain, wlplan_problem, state)
        bool_feature_eval = bool_eval_state(fg, features, wlplan_domain, wlplan_problem, state)
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
            succs_evals = [eval_state(fg, features, wlplan_domain, wlplan_problem, succ) for succ in succs]
            log.debug(f"succs_evals: {succs_evals}")
            succs_diffs = {eval_state_diff(feature_eval, succ_eval) for succ_eval in succs_evals}
            log.debug(f'succs_diffs:\n{"\n".join([", ".join([str(d) for d in ds]) for ds in succs_diffs])}')
            ok = True
            for constraint in enabled_constraints:
                if constraint.effs & succs_diffs:
                    log.info(f"Constraint {constraint} violated!")
                    ok = False
                    break
            bool_succs_evals = [bool_eval_state(fg, features, wlplan_domain, wlplan_problem, succ) for succ in succs]
            for bool_succs_eval in bool_succs_evals:
                for state_constraint in policy.state_constraints:
                    violated = True
                    for feature_str, cond in state_constraint.conds.items():
                        if cond != bool_succs_eval[feature_str]:
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
