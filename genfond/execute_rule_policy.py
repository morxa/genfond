from .policy import Effect, PolicyType
from .generate_rule_policy import feature_eval_to_cond
from .feature_generator import construct_vocabulary_info, construct_instance_info, get_augmented_state, _get_state_from_goal
from .ground import ground
from dlplan.core import SyntacticElementFactory, State
from .state_space_generator import check_formula, apply_action_effects
import random

import logging

log = logging.getLogger(__name__)


def eval_state(instance, mapping, features, domain, problem, state, config):
    try:
        fstate = State(-1, instance, [mapping[predicate] for predicate in get_augmented_state(problem, state, config)])
    except KeyError as e:
        log.critical(f'Cannot find predicate in mapping {"\n".join(f"{k}: {v}" for k, v in mapping.items())}: {e}')
        raise
    feature_eval = dict()
    for fstring, feature in features.items():
        feature_eval[fstring] = feature.evaluate(fstate)
    return feature_eval


def bool_eval_state(instance, mapping, features, domain, problem, state, config):
    feature_eval = eval_state(instance, mapping, features, domain, problem, state, config)
    log.debug(f'feature eval: {feature_eval}')
    bool_feature_eval = dict()
    for feature, eval in feature_eval.items():
        bool_feature_eval[feature] = feature_eval_to_cond(feature, eval)
    return bool_feature_eval


def eval_state_diff(state, succ):
    log.debug(f'eval state diff: {state} -> {succ}')
    diff = set()
    for feature in succ:
        if state[feature] == succ[feature]:
            continue
        elif feature.startswith("b_"):
            if not state[feature] and succ[feature]:
                diff.add((feature, Effect.SET))
            elif state[feature] and not succ[feature]:
                diff.add((feature, Effect.UNSET))
            else:
                raise RuntimeError(f'Inconsistent state transition: {state} -> {succ}')
        elif feature.startswith("n_"):
            if state[feature] < succ[feature]:
                diff.add((feature, Effect.INCREASE))
            elif state[feature] > succ[feature]:
                diff.add((feature, Effect.DECREASE))
            else:
                raise RuntimeError(f'Inconsistent state transition: {state} -> {succ}')
        else:
            raise ValueError(f'Unknown feature type: {feature}')
    return frozenset(diff)


def state_satisfies_rule_conds(bool_feature_eval, rule_conds):
    for feature, cond in rule_conds.items():
        if cond != bool_feature_eval[feature]:
            return False
    return True


def get_next_state(states, action):
    return random.choice([state for state in states])


def action_string(action):
    return f'{action.name}({",".join([str(p) for p in action.parameters])})'


def state_string(state):
    return ",".join([str(p) for p in state])


def execute_rule_policy(domain, problem, policy, config):
    log.info(
        f'Executing policy:\n{policy}\nin {domain.name} for problem {problem.name} with features {policy.features}')
    vocabulary = construct_vocabulary_info(domain)
    factory = SyntacticElementFactory(vocabulary)
    instance, mapping = construct_instance_info(vocabulary, domain, problem, 0)
    features = dict()
    for feature in policy.features:
        if feature.startswith("b_"):
            features[feature] = factory.parse_boolean(feature)
        elif feature.startswith("n_"):
            features[feature] = factory.parse_numerical(feature)
        else:
            raise ValueError(f'Unknown feature type: {feature}')
    # TODO _get_state_from_goal is internal
    goal_state = _get_state_from_goal(problem.goal)
    log.debug("Grounding actions...")
    grounded_actions = ground(domain, problem)
    log.debug("Grounding actions done.")
    state = problem.init
    num_steps = 0
    actions_taken = []
    max_steps = config['policy_steps']
    while not check_formula(state, problem.goal) and (max_steps <= 0 or num_steps < max_steps):
        log.info(f'New state: {",".join([str(p) for p in state])}')
        feature_eval = eval_state(instance, mapping, features, domain, problem, state, config)
        bool_feature_eval = bool_eval_state(instance, mapping, features, domain, problem, state, config)
        enabled_rules = {rule for rule in policy.rules if state_satisfies_rule_conds(bool_feature_eval, rule.conds)}
        log.debug('Enabled rules: {}'.format(",  ".join([str(r) for r in enabled_rules])))
        enabled_constraints = {
            constraint
            for constraint in policy.constraints if state_satisfies_rule_conds(bool_feature_eval, constraint.conds)
        }
        log.debug('Enabled constraints: {}'.format(",  ".join([str(c) for c in enabled_constraints])))
        if not enabled_rules:
            log.error("No rule enabled!")
            raise RuntimeError("No rule enabled!")
        found_rule = False
        log.debug('Enabled actions: {}'.format(", ".join(
            [action_string(a) for a in grounded_actions if check_formula(state, a.precondition)])))
        for action in sorted(grounded_actions, key=lambda _: random.random()):
            if not check_formula(state, action.precondition):
                continue
            succs = apply_action_effects(state, action)
            log.debug('Action {} has {} successors: {}'.format(action_string(action), len(succs),
                                                               "; ".join([state_string(s) for s in succs])))
            succs_evals = [eval_state(instance, mapping, features, domain, problem, succ, config) for succ in succs]
            log.debug(f'succs_evals: {succs_evals}')
            succs_diffs = {eval_state_diff(feature_eval, succ_eval) for succ_eval in succs_evals}
            log.debug(f'succs_diffs:\n{"\n".join([", ".join([str(d) for d in ds]) for ds in succs_diffs])}')
            ok = True
            for constraint in enabled_constraints:
                if constraint.effs & succs_diffs:
                    log.info(f'Constraint {constraint} violated!')
                    ok = False
                    break
            bool_succs_evals = [
                bool_eval_state(instance, mapping, features, domain, problem, succ, config) for succ in succs
            ]
            for bool_succs_eval in bool_succs_evals:
                for state_constraint in policy.state_constraints:
                    violated = True
                    for feature, cond in state_constraint.conds.items():
                        if cond != bool_succs_eval[feature]:
                            violated = False
                            break
                    if violated:
                        log.info(f'State constraint {state_constraint} violated!')
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for rule in enabled_rules:
                log.debug(f'Checking rule {rule}')
                if rule.effs == succs_diffs or policy.type == PolicyType.CONSTRAINED and rule.effs & succs_diffs:
                    found_rule = True
                    log.info(f'Found matching rule:\n{rule}')
                    log.info(f'Applying action {action_string(action)}')
                    state = get_next_state(succs, action)
                    num_steps += 1
                    actions_taken.append(action_string(action))
                    break
            if found_rule:
                break
        if not found_rule:
            log.error(f'No matching rule found for diff {succs_diffs}!')
            log.error('Enabled rules:\n  {}'.format("\n  ".join([str(r) for r in enabled_rules])))
            raise RuntimeError('No matching rule found!')
    if not check_formula(state, problem.goal):
        log.error('Goal not reached!')
        raise RuntimeError('Goal not reached!')
    log.info('Goal reached!')
    return actions_taken
