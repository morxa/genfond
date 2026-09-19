import logging
import random
import time
from typing import Any, Collection, Iterator, Optional

import dlplan.core
from dlplan.core import Concept, ConceptDenotation, InstanceInfo, Role, RoleDenotation, SyntacticElementFactory
from pddl.action import Action
from pddl.core import Domain, Problem
from pddl.logic.base import And, Formula, Not
from pddl.logic.predicates import Predicate

from genfond.config_handler import ConfigHandler
from genfond.datalog_policy import DatalogPolicy, DatalogPolicyRule
from genfond.ground import action_string, state_string

from .execute_rule_policy import (
    CycleError,
    ExecutionTimeout,
    NoActionError,
    PolicyExecutionError,
    _get_dlplan_state,
    state_satisfies_rule_conds,
)
from .feature_generator import (
    Feature,
    _get_state_from_goal,
    construct_instance_info,
    construct_vocabulary_info,
    get_aparam_predicate,
)
from .generate_rule_policy import feature_eval_to_cond
from .ground import ground
from .rule_policy import Cond
from .state_space_generator import State, apply_action_effects, check_formula

log = logging.getLogger("genfond.execution.datalog")


def get_next_state(states: Collection[State]) -> State:
    # Sorted before the draw: `states` is a `set[State]` and `State` is a `frozenset` of `pddl`
    # `Predicate`s, whose hash is address-dependent (see `ground._stable_constants`), so the
    # unsorted order differs between two identically seeded processes. A single successor -- the
    # deterministic case, i.e. every step of a deterministic domain -- admits only one order, so
    # the sort key (which stringifies the whole state) is skipped there. `random.choice` still
    # runs, and still draws from the RNG, so the draw sequence is the same either way.
    candidates = list(states)
    if len(candidates) > 1:
        candidates.sort(key=state_string)
    return random.choice(candidates)


def eval_concepts(concepts: dict[str, Concept], fstate: dlplan.core.State) -> dict[str, ConceptDenotation]:
    return {concept_string: concept.evaluate(fstate) for concept_string, concept in concepts.items()}


def eval_roles(roles: dict[str, Role], fstate: dlplan.core.State) -> dict[str, RoleDenotation]:
    return {role_string: role.evaluate(fstate) for role_string, role in roles.items()}


def bool_eval(features: dict[str, Feature], fstate: dlplan.core.State) -> dict[str, Cond]:
    """`execute_rule_policy.bool_eval_state` for an already-constructed dlplan state.

    The plain (goal-augmented, action-free) dlplan state that the concepts and roles are
    evaluated on is the same one the features need, so the executor builds it once per step and
    shares it instead of re-deriving it from the `pddl` state three times. `logger=log` for the
    same reason as in `bool_eval_state`: it keeps the per-condition DEBUG line on
    `genfond.execution.*`, which the `log:` map silences by default.
    """
    return {
        name: feature_eval_to_cond(name, feature.evaluate(fstate), logger=log) for name, feature in features.items()
    }


def _flatten_precondition(formula: Formula) -> Optional[tuple[tuple[Predicate, ...], tuple[Predicate, ...]]]:
    """Split a conjunction of literals into its positive and negative atoms.

    Returns `None` for any precondition that is not a (possibly nested) conjunction of atoms and
    negated atoms -- numeric comparisons, disjunctions, quantifiers -- and the caller then falls
    back to the general `check_formula`. For the flattenable case, which is every STRIPS action,
    applicability becomes two membership scans over the state instead of a recursive walk over
    the formula, and binding search evaluates it once per candidate ground action.
    """
    pos: list[Predicate] = []
    neg: list[Predicate] = []

    def walk(f: Formula) -> bool:
        if isinstance(f, And):
            return all(walk(operand) for operand in f.operands)
        if isinstance(f, Predicate):
            pos.append(f)
            return True
        if isinstance(f, Not) and isinstance(f.argument, Predicate):
            neg.append(f.argument)
            return True
        return False

    if not walk(formula):
        return None
    return tuple(pos), tuple(neg)


class _GroundActionIndex:
    """The ground actions of one action name, indexed for binding search.

    A datalog rule head `name(P, Q, ...)` is instantiated by choosing one object per parameter and
    looking the resulting tuple up among the ground actions. Enumerating the full object product
    and looking every tuple up is what made execution scale with |objects|^arity: for
    `tighten_nut` on a 24-object spanner instance that is 24^4 = 331k tuples per rule per step,
    of which 136 name a ground action at all.

    `positions[i]` is the set of object names occurring at parameter position `i` of *some* ground
    action, and `trie` maps a parameter tuple, position by position, onto its leaf
    `(action, precondition literals)` entry. Filtering each parameter's candidate list by
    `positions[i]` and then walking `trie` enumerates exactly the tuples that name a ground
    action, in the same order the full product visited them: the tuples skipped are precisely
    those the old `ground_actions.get(...)` lookup rejected.
    """

    __slots__ = ("arity", "positions", "trie")

    def __init__(self, actions: list[Action], arity: int) -> None:
        self.arity = arity
        self.positions: list[set[str]] = [set() for _ in range(arity)]
        trie: Any = {} if arity else None
        for action in actions:
            params = tuple(str(p.name) for p in action.parameters)
            for i, name in enumerate(params):
                self.positions[i].add(name)
            entry = (action, _flatten_precondition(action.precondition))
            if arity == 0:
                trie = entry
                continue
            node = trie
            for name in params[:-1]:
                node = node.setdefault(name, {})
            node[params[-1]] = entry
        self.trie = trie

    def bindings(self, objects: list[list[str]], state: State) -> Iterator[Action]:
        """The ground actions of this name instantiating `objects` that are applicable in `state`.

        `objects[i]` is the (already shuffled) candidate list for parameter `i`; the actions are
        yielded in the lexicographic order of those lists, i.e. the order `itertools.product`
        would have produced them in.
        """
        if len(objects) != self.arity or self.trie is None:
            return
        filtered = [[o for o in objects[i] if o in self.positions[i]] for i in range(self.arity)]
        if any(not candidates for candidates in filtered):
            return
        yield from self._walk(filtered, 0, self.trie, state)

    def _walk(self, objects: list[list[str]], depth: int, node: Any, state: State) -> Iterator[Action]:
        if depth == self.arity:
            action, literals = node
            if literals is None:
                if check_formula(state, action.precondition):
                    yield action
            else:
                pos, neg = literals
                if all(atom in state for atom in pos) and not any(atom in state for atom in neg):
                    yield action
            return
        for name in objects[depth]:
            child = node.get(name)
            if child is not None:
                yield from self._walk(objects, depth + 1, child, state)


def _build_action_index(domain: Domain, problem: Problem) -> dict[str, _GroundActionIndex]:
    arity = {action.name: len(action.parameters) for action in domain.actions}
    by_name: dict[str, list[Action]] = {name: [] for name in arity}
    for action in ground(domain, problem):
        by_name[action.name].append(action)
    return {name: _GroundActionIndex(actions, arity[name]) for name, actions in by_name.items()}


def _parse_policy_elements(
    datalog_policy: DatalogPolicy, factory: SyntacticElementFactory
) -> tuple[dict[str, Feature], dict[str, Concept], dict[str, Role]]:
    features: dict[str, Feature] = dict()
    concepts: dict[str, Concept] = dict()
    roles: dict[str, Role] = dict()
    for rule in datalog_policy.rules:
        for cond in (
            rule.conds
            | rule.state_aug_conds
            | rule.param_aug_conds
            | {cond[0]: None for cond in rule.param_diff_conds}
        ):
            if cond.startswith("b_"):
                features[cond] = factory.parse_boolean(cond)
            elif cond.startswith("n_"):
                features[cond] = factory.parse_numerical(cond)
            else:
                raise ValueError(f"Unknown feature type: {cond}")
        for rule_concepts in rule.concepts_by_parameter.values():
            for concept in rule_concepts:
                concepts[concept] = factory.parse_concept(concept)
        for rule_roles in rule.roles_by_parameter.values():
            for role in rule_roles:
                roles[role] = factory.parse_role(role)
    return features, concepts, roles


def _role_checks(rule: DatalogPolicyRule) -> list[tuple[int, int, str]]:
    """The rule's role conditions as (parameter index, parameter index, role) triples.

    Resolved once per policy rather than once per candidate binding: the rule stores them keyed by
    parameter *name*, and the executor used to call `rule.parameters.index(...)` for both ends of
    every role of every candidate. All of them have to hold, so the order is irrelevant; sorting
    it only removes a dependence on frozenset iteration order.
    """
    checks = []
    for role_params, rule_roles in sorted(rule.roles_by_parameter.items()):
        index_0 = rule.parameters.index(role_params[0])
        index_1 = rule.parameters.index(role_params[1])
        for role in sorted(rule_roles):
            checks.append((index_0, index_1, role))
    return checks


def _param_augmented_dlplan_state(
    instance: InstanceInfo,
    mapping: dict,
    state: State,
    goal_atoms: Collection[Predicate],
    param: Any,
    cache: dict[Any, dlplan.core.State],
) -> dlplan.core.State:
    """`feature_generator.get_param_augmented_state` lifted to dlplan, memoised within a step.

    The augmented state depends only on the current state and the marked parameter, so the cache
    key is the parameter: within one step, a rule with several parameter conditions -- or several
    candidate bindings over the same object -- reuses it instead of rebuilding the goal-augmented
    atom set and re-mapping every atom of it.
    """
    key = ("param", str(param))
    cached = cache.get(key)
    if cached is not None:
        return cached
    atoms = set(state)
    atoms.update(goal_atoms)
    atoms.add(get_aparam_predicate(0, param))
    aug_state = dlplan.core.State(-1, instance, [mapping[atom] for atom in atoms])
    cache[key] = aug_state
    return aug_state


def execute_datalog_policy(
    domain: Domain,
    problem: Problem,
    datalog_policy: DatalogPolicy,
    config: ConfigHandler,
    time_limit: Optional[float] = None,
    out_actions: Optional[list[Action]] = None,
) -> list[str]:
    """Run `datalog_policy` from `problem.init` and return the action strings it applied.

    `out_actions`, when given, is additionally filled with the *ground actions* themselves, in
    the order they were applied -- the same trajectory the returned strings describe, but in a
    form a `pddl.core.Plan` can be rebuilt from (see `problem_iterator.plan_from_actions`).
    Existing callers pass nothing and are unaffected. The list is appended to as execution
    proceeds, so on a failure it holds the prefix taken before the failure; only a call that
    returns normally has a trajectory that actually reaches the goal.

    The RNG is drawn from in a fixed sequence per step -- one `random.sample` over the rules,
    then per rule one `random.shuffle` per parameter reached before the first empty candidate
    set, and finally one `random.choice` over the successors of the chosen action -- which is
    what makes a seeded execution reproducible. Every optimisation below only skips candidate
    bindings that the conditions would have rejected anyway, so none of them moves that sequence.
    """
    if log.isEnabledFor(logging.INFO):
        log.info(f"Executing policy:\n{datalog_policy}\nin {domain.name} for problem {problem.name}")
    # See execute_rule_policy.execute_rule_policy for why this is a per-step monotonic-clock
    # check rather than a signal-based interrupt or a policy_steps substitute.
    deadline = time.monotonic() + time_limit if time_limit else None

    vocabulary = construct_vocabulary_info(domain, config)
    factory = SyntacticElementFactory(vocabulary)
    instance, mapping = construct_instance_info(vocabulary, domain, problem, 0, config)
    object_id_to_name = {o.get_index(): o.get_name() for o in instance.get_objects()}
    object_name_to_id = {name: index for index, name in object_id_to_name.items()}
    all_object_ids = frozenset(object_id_to_name)

    features, concepts, roles = _parse_policy_elements(datalog_policy, factory)
    log.info("Grounding actions...")
    action_index = _build_action_index(domain, problem)
    log.info("Grounding actions done!")

    # The goal atoms and the action-free config are the same at every step; deriving them per
    # feature evaluation was a constant multiplier on the per-step cost.
    goal_atoms = _get_state_from_goal(problem.goal)
    plain_config = config | {"include_actions": False}
    rules_in_order = sorted(datalog_policy.rules, key=repr)
    role_checks = {rule: _role_checks(rule) for rule in rules_in_order}

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
        # Hoisted out of the f-strings: at the default execution log level (CRITICAL) every one
        # of these lines was formatted and thrown away, and the ones inside the binding loop were
        # formatted once per candidate tuple.
        debug = log.isEnabledFor(logging.DEBUG)
        info = log.isEnabledFor(logging.INFO)
        if debug:
            log.debug("-" * 80)
        if info:
            log.info(f"New state: {state_string(state)}")
        found_rule = False

        fstate = _get_dlplan_state(instance, mapping, problem, state, plain_config, None)
        concepts_eval = {name: frozenset(den.to_vector()) for name, den in eval_concepts(concepts, fstate).items()}
        roles_eval = {name: frozenset(den.to_vector()) for name, den in eval_roles(roles, fstate).items()}
        feature_eval = bool_eval(features, fstate)
        aug_state_cache: dict[Any, dlplan.core.State] = dict()

        for rule in random.sample(rules_in_order, len(rules_in_order)):
            if debug:
                log.debug(f"Checking rule: {rule}")
            if not state_satisfies_rule_conds(feature_eval, rule.conds):
                if debug:
                    log.debug("... Rule conditions not satisfied!")
                continue
            if debug:
                log.debug("... Rule conditions satisfied!")
            objects: list[list[str]] = [[] for _ in range(len(rule.parameters))]

            for param_index, parameter in enumerate(rule.parameters):
                valid_objects = all_object_ids
                for concept in rule.concepts_by_parameter[parameter]:
                    valid_objects &= concepts_eval[concept]
                    if not valid_objects:
                        break

                objects[param_index] = [object_id_to_name[i] for i in sorted(valid_objects)]
                # Draws len-1 values from the RNG. Everything downstream only *filters* this
                # list, preserving its order, so the candidate order -- and hence the action
                # chosen -- is the one the unoptimised executor enumerated.
                random.shuffle(objects[param_index])
                if not valid_objects:
                    break

            if debug:
                log.debug(f"... Found valid objects {objects}")
            if [] in objects:
                if debug:
                    log.debug("... Rule not applicable! Not all objects found!")
                continue

            head_index = action_index.get(rule.name)
            action = None
            for candidate in head_index.bindings(objects, state) if head_index else ():
                parameter_names = [str(p.name) for p in candidate.parameters]
                if debug:
                    log.debug(f"... Checking rule with object combination {tuple(parameter_names)}")
                valid = True
                for index_0, index_1, role in role_checks[rule]:
                    pair = (object_name_to_id[parameter_names[index_0]], object_name_to_id[parameter_names[index_1]])
                    if pair not in roles_eval[role]:
                        if debug:
                            log.debug(
                                f"... Role {role} not satisfied for {parameter_names[index_0]} "
                                f"and {parameter_names[index_1]}"
                            )
                        valid = False
                        break
                if not valid:
                    continue
                action = candidate

                if rule.state_aug_conds:
                    # Hoisted out of the condition loop: every condition evaluated the *same*
                    # augmented state, rebuilding it and re-evaluating every feature each time.
                    aug_key = ("action", candidate.name, tuple(parameter_names))
                    aug_state = aug_state_cache.get(aug_key)
                    if aug_state is None:
                        aug_state = _get_dlplan_state(instance, mapping, problem, state, config, candidate)
                        aug_state_cache[aug_key] = aug_state
                    aug_feature_eval = bool_eval(features, aug_state)
                    for cond, val in rule.state_aug_conds.items():
                        if aug_feature_eval[cond] != val:
                            if debug:
                                log.debug(
                                    f"... Rule not applicable! Augmented state does not satisfy condition {cond}"
                                    f" (valuation is {aug_feature_eval[cond]} instead of {val})"
                                )
                            action = None
                            break
                if not action:
                    continue
                for cond, pval in rule.param_aug_conds.items():
                    aug_param_index, val = pval
                    param = candidate.parameters[aug_param_index]
                    param_state = _param_augmented_dlplan_state(
                        instance, mapping, state, goal_atoms, param, aug_state_cache
                    )
                    feval = feature_eval_to_cond(cond, features[cond].evaluate(param_state), logger=log)
                    if feval != val:
                        if debug:
                            log.debug(
                                f"... Rule not applicable! Augmented state does not satisfy condition {cond} on "
                                f"{param} (valuation is {feval} instead of {val})"
                            )
                        action = None
                        break
                if not action:
                    continue
                if debug:
                    log.debug(f"... Checking {len(rule.param_diff_conds)} diff conditions")
                for feature, param1, param2, diff in rule.param_diff_conds:
                    assert diff in [-1, 0, 1], f"Invalid diff value: {diff}"
                    aug_state1 = _param_augmented_dlplan_state(
                        instance, mapping, state, goal_atoms, candidate.parameters[param1], aug_state_cache
                    )
                    aug_state2 = _param_augmented_dlplan_state(
                        instance, mapping, state, goal_atoms, candidate.parameters[param2], aug_state_cache
                    )
                    eval1 = features[feature].evaluate(aug_state1)
                    eval2 = features[feature].evaluate(aug_state2)
                    eval_diff = 1 if eval1 < eval2 else -1 if eval1 > eval2 else 0
                    if eval_diff != diff:
                        if debug:
                            log.debug(
                                f"... Rule not applicable! Augmented state does not satisfy condition {feature} "
                                f"(valuation is {eval_diff} instead of {diff})"
                            )
                        action = None
                        break

                if action:
                    break

            if not action:
                if debug:
                    log.debug("... Rule not applicable! No matching action found!")
                continue

            if info:
                log.info(f"... Found action {action_string(action)}")
                log.info(f"{rule}")
                log.info(f"Applying action {action_string(action)}")
            found_rule = True
            new_state = get_next_state(apply_action_effects(state, action))
            trace[state] = new_state
            state = new_state
            num_steps += 1
            actions_taken.append(action_string(action))
            if out_actions is not None:
                out_actions.append(action)
            break

        if not found_rule:
            log.error(f"No matching rule found for state {state_string(state)}!")
            raise NoActionError(trace, state)

    if not check_formula(state, problem.goal):
        log.error("Goal not reached!")
        raise PolicyExecutionError("Goal not reached!")

    log.info("Goal reached!")
    return actions_taken
