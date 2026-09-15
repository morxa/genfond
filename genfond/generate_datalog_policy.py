import logging
import re
from typing import Any, Iterable, Iterator, Optional, Sequence

from genfond.action_signatures import ActionSignature
from genfond.datalog_policy import (
    RULE_VARS,
    DatalogPolicy,
    DatalogPolicyRule,
    split_action_string,
)
from genfond.rule_policy import Cond, Effect

log = logging.getLogger("genfond.generation.datalog")


def eval_to_cond(f: str, v: int) -> Cond:
    if f.startswith("b_"):
        if v == 1:
            return Cond.TRUE
        elif v == 0:
            return Cond.FALSE
        else:
            raise ValueError(f"Unknown value {v}")
    elif f.startswith("n_"):
        if v == 1:
            return Cond.POSITIVE
        elif v == 0:
            return Cond.ZERO
        else:
            raise ValueError(f"Unknown value {v}")
    else:
        raise ValueError(f"Unknown value {v}")


def _bitmask(items: Iterable[Any], index: dict[Any, int]) -> int:
    mask = 0
    for item in items:
        mask |= 1 << index.setdefault(item, len(index))
    return mask


def _set_bits(mask: int) -> Iterator[int]:
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length() - 1
        mask ^= lowest


def generate_datalog_policy_from_signatures(
    solution: dict[str, Any], signatures: Sequence[ActionSignature]
) -> DatalogPolicy:
    """Build the policy from a model of solve_datalog_sig.lp.

    That program collapses the separation layer onto action signature classes, so a rule is
    produced per good signature rather than per (instance, state, good action) triple.

    The conditions of a rule are reconstructed here rather than shown by the solve program,
    which would need a rule per (class pair, element). The model supplies the good classes
    (sig_action/2), the bad ones (bad_sig/1) and the selection; everything else is a property
    of the signature classes, which the caller passes in. The reconstruction mirrors the layered
    #show rules the program used to carry: a bad class already separated by a selected feature
    contributes nothing, one separated by a selected concept contributes only concept
    conditions, and the rest contribute role conditions.
    """
    heads: dict[int, str] = dict()
    for signature_id, action in solution.get("sig_action", []):
        heads.setdefault(signature_id, action)
    bad_ids = sorted(solution.get("bad_sig", set()))
    selected_features = {feature.strip('"') for feature in solution.get("f_selected", set())}
    # `name` is the always-selected identity concept; no signature ever carries it.
    selected_concepts = {concept.strip('"') for concept in solution.get("c_selected", set())} - {"name"}
    selected_roles = {role.strip('"') for role in solution.get("r_selected", set())}

    # A feature is attached as a condition iff it is selected and not constant across all
    # classes -- exactly what the program's sig_f_dist/3 derived (it ranged over all classes,
    # not only over those of the same action name).
    seen_value: dict[str, int] = dict()
    varying_features: set[str] = set()
    for signature in signatures:
        for feature, value in signature.bools:
            if seen_value.setdefault(feature, value) != value:
                varying_features.add(feature)
    cond_features = sorted(selected_features & varying_features)

    relevant = sorted(set(heads) | set(bad_ids))
    concept_index: dict[Any, int] = dict()
    role_index: dict[Any, int] = dict()
    feature_key: dict[int, tuple[tuple[str, int], ...]] = dict()
    concept_mask: dict[int, int] = dict()
    role_mask: dict[int, int] = dict()
    for signature_id in relevant:
        signature = signatures[signature_id]
        feature_key[signature_id] = tuple(
            sorted((feature, value) for feature, value in signature.bools if feature in selected_features)
        )
        concept_mask[signature_id] = _bitmask(
            ((concept, position) for concept, position in signature.concepts if concept in selected_concepts),
            concept_index,
        )
        role_mask[signature_id] = _bitmask(
            ((role, position1, position2) for role, position1, position2 in signature.roles if role in selected_roles),
            role_index,
        )
    concept_terms = [term for term, _ in sorted(concept_index.items(), key=lambda item: item[1])]
    role_terms = [term for term, _ in sorted(role_index.items(), key=lambda item: item[1])]

    rules = set()
    for signature_id, action in heads.items():
        signature = signatures[signature_id]
        values = dict(signature.bools)
        conds = {feature: eval_to_cond(feature, values[feature]) for feature in cond_features if feature in values}
        concepts_pos = concepts_neg = roles_pos = roles_neg = 0
        for other_id in bad_ids:
            other = signatures[other_id]
            if other.name != signature.name or other.arity != signature.arity:
                continue
            if feature_key[signature_id] != feature_key[other_id]:
                continue
            differing = concept_mask[signature_id] ^ concept_mask[other_id]
            if differing:
                concepts_pos |= concept_mask[signature_id] & differing
                concepts_neg |= concept_mask[other_id] & differing
                continue
            differing = role_mask[signature_id] ^ role_mask[other_id]
            roles_pos |= role_mask[signature_id] & differing
            roles_neg |= role_mask[other_id] & differing
        concepts = {(position, concept) for concept, position in (concept_terms[b] for b in _set_bits(concepts_pos))}
        concepts |= {
            (position, f"c_not({concept})")
            for concept, position in (concept_terms[b] for b in _set_bits(concepts_neg))
        }
        roles = {
            (position1, position2, role)
            for role, position1, position2 in (role_terms[b] for b in _set_bits(roles_pos))
        }
        roles |= {
            (position1, position2, f"r_not({role})")
            for role, position1, position2 in (role_terms[b] for b in _set_bits(roles_neg))
        }
        name, parameters = split_action_string(action)
        variables = RULE_VARS[: len(parameters)]
        rules.add(
            DatalogPolicyRule(
                f'{name}({",".join(variables)})',
                concepts=[(variables[index], concept) for index, concept in sorted(concepts)],
                roles=[(variables[index1], variables[index2], role) for index1, index2, role in sorted(roles)],
                conds=conds,
            )
        )
    # Logged at INFO (not DEBUG): the good-signature count is the quantity
    # minimize_good_signatures trades off against feature cost, so it needs to be visible by
    # default to judge the effect of that setting on a run.
    log.info(f"Generated {len(rules)} rule(s) from {len(heads)} good signature(s), {len(bad_ids)} bad signature(s)")
    return DatalogPolicy(list(rules), cost=solution["cost"])


def generate_datalog_policy(
    solution: dict[str, Any], signatures: Optional[Sequence[ActionSignature]] = None
) -> DatalogPolicy:
    if "sig_action" in solution:
        assert signatures is not None, "the signature-quotiented encoding needs the signature classes"
        return generate_datalog_policy_from_signatures(solution, signatures)
    # log.info(
    #     f'Generating policy from solution with {len(solution["good_action"])}/{len(solution.get("trans", []) or "?")} good actions,'
    #     f' {len(solution.get("f_distinguished", []))} distinguished features,'
    #     f' {len(solution.get("c_distinguished", []))} distinguished concepts,'
    #     f' {len(solution.get("r_distinguished", []))} distinguished roles')
    log.debug(f'goals: {solution.get("goal", [])}')
    log.debug(f'safe states: {sorted(solution.get("safe_state", []))}')
    log.debug(f'good_trans: {sorted(solution.get("good_trans", []))}')
    args_to_vars = dict()
    conds: dict[tuple[int, int, str], dict[str, Any]] = dict()
    for instance, state, action in solution.get("good_action", []):
        log.debug(f"Good action {action} in state {state} of instance {instance}")
        name, parameters = split_action_string(action)
        arg_to_var = dict()
        vars = RULE_VARS.copy()
        for i, parameter in enumerate(parameters):
            if parameter not in arg_to_var:
                arg_to_var[i] = vars.pop(0)
        args_to_vars[(instance, state, action)] = arg_to_var
    bool_eval_dict = dict()
    for i, s, f, v in solution.get("bool_eval", []):
        bool_eval_dict[(i, s, f)] = v
    aug_bool_eval_dict = dict()
    state_aug_bool_eval_dict = dict()
    for i, s, a, f, v in solution.get("state_aug_bool_eval", []):
        state_aug_bool_eval_dict[(i, s, a, f)] = v
    for i, s, a, p, f, v in solution.get("aug_bool_eval", []):
        aug_bool_eval_dict[(i, s, a, p, f)] = v
    rules = set()
    dist_features: dict[tuple[int, int], list[str]] = dict()
    state_aug_dist_features: dict[tuple[int, int, str], list[str]] = dict()
    param_aug_dist_features: dict[tuple[int, int, str], list[tuple[str, int]]] = dict()
    for instance, state, _, _, feature in solution.get("f_distinguished", []):
        dist_features.setdefault((instance, state), []).append(feature)
    for instance, state, action, _, _, _, feature in solution.get("state_aug_dist", []):
        state_aug_dist_features.setdefault((instance, state, action), []).append(feature)
    for instance, state, action, param, feature in solution.get("param_aug_dist", []):
        param_aug_dist_features.setdefault((instance, state, action), []).append((feature, param))
    diff_conds: dict[tuple[int, int, str], list[tuple[str, int, int, int]]] = dict()
    for instance, state, action, param1, param2, feature, diff in solution.get(f"aug_d2", []):
        diff_conds.setdefault((instance, state, action), []).append((feature, param1, param2, diff))
    state_conds = dict()
    for instance, state, action in solution.get("good_action", []):
        state_cond: dict[str, Cond] = dict()
        state_aug_cond: dict[str, Cond] = dict()
        param_aug_cond: dict[str, tuple[int, Cond]] = dict()
        for f in dist_features.get((instance, state), []):
            v = bool_eval_dict[(instance, state, f)]
            log.debug(f"Adding state condition {f}={v}")
            state_cond[f] = eval_to_cond(f, v)
        state_conds[(instance, state, action)] = state_cond
        for f in state_aug_dist_features.get((instance, state, action), []):
            v = state_aug_bool_eval_dict[(instance, state, action, f)]
            log.debug(f"Adding state augmented condition {f}={v}")
            state_aug_cond[f] = eval_to_cond(f, v)
        for f, p in param_aug_dist_features.get((instance, state, action), []):
            v = aug_bool_eval_dict[(instance, state, action, p, f)]
            log.debug(f"({instance}, {state}, {action}): Adding augmented condition {f}={v} for param {p}")
            param_aug_cond[f] = (p, eval_to_cond(f, v))
        diff_cond: list[tuple[str, int, int, int]] = []
        for f, param1, param2, v in diff_conds.get((instance, state, action), []):
            log.debug(f"({instance}, {state}, {action}): Adding diff condition {f}({param1},{param2})={v}")
            diff_cond.append((f, param1, param2, v))
        conds[(instance, state, action)] = {
            "concepts": [],
            "roles": [],
            "state_aug_conds": state_aug_cond,
            "param_aug_conds": param_aug_cond,
            "diff_conds": diff_cond,
        }
    log.debug(f"state_conds: {state_conds}")
    for instance, state, action, _, _, _, concept, pos, argnum in solution.get("c_distinguished", []):
        action = action.strip('"')
        concept = concept.strip('"')
        argnum = int(argnum)
        negated = pos == "neg"
        if concept == "name":
            continue
        if negated:
            concept = f"c_not({concept})"
        var = args_to_vars[(instance, state, action)][argnum]
        conds[(instance, state, action)]["concepts"].append((var, concept))
    for instance, state, action, _, _, _, role, pos, argnum1, argnum2 in solution.get("r_distinguished", []):
        action = action.strip('"')
        role = role.strip('"')
        argnum1 = int(argnum1)
        argnum2 = int(argnum2)
        negated = pos == "neg"
        if negated:
            role = f"r_not({role})"
        var1 = args_to_vars[(instance, state, action)][argnum1]
        var2 = args_to_vars[(instance, state, action)][argnum2]
        conds[(instance, state, action)]["roles"].append((var1, var2, role))
    for key, cond_dict in conds.items():
        action = key[2]
        action_name, _ = split_action_string(action)
        args = ",".join(args_to_vars[key].values())
        action = f"{action_name}({args})"
        rule = DatalogPolicyRule(
            action,
            concepts=cond_dict["concepts"],
            roles=cond_dict["roles"],
            conds=state_conds[key],
            state_aug_conds=cond_dict["state_aug_conds"],
            param_aug_conds=cond_dict["param_aug_conds"],
            param_diff_conds=cond_dict["diff_conds"],
        )
        rules.add(rule)
    return DatalogPolicy(list(rules), cost=solution["cost"])
