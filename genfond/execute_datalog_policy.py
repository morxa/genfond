import itertools
import logging
import random
from typing import Collection, Optional

from dlplan.core import (
    Concept,
    ConceptDenotation,
    InstanceInfo,
    Role,
    RoleDenotation,
    SyntacticElementFactory,
)
from pddl.action import Action
from pddl.core import Domain, Problem

from genfond.config_handler import ConfigHandler
from genfond.datalog_policy import DatalogPolicy

from .execute_rule_policy_dlplan import (
    CycleError,
    NoActionError,
    PolicyExecutionError,
    _get_dlplan_state,
    bool_eval_state,
    state_satisfies_rule_conds,
)
from .feature_generator_dlplan import (
    construct_instance_info,
    construct_vocabulary_info,
)
from .generate_rule_policy import feature_eval_to_cond
from .ground import ground
from .state_space_generator import State, apply_action_effects, check_formula

log = logging.getLogger("genfond.execution.datalog")


def get_next_state(states: Collection[State]) -> State:
    return random.choice([state for state in states])


def action_string(action: Action) -> str:
    return f'{action.name}({",".join([str(p) for p in action.parameters])})'


def state_string(state) -> str:
    return ",".join([str(p) for p in state])


def eval_concepts(
    instance: InstanceInfo,
    mapping: dict,
    concepts: dict[str, Concept],
    problem: Problem,
    state: State,
    config: dict,
    action: Optional[Action] = None,
) -> dict[str, ConceptDenotation]:
    fstate = _get_dlplan_state(instance, mapping, problem, state, config, action)
    eval = dict()
    for concept_string, concept in concepts.items():
        eval[concept_string] = concept.evaluate(fstate)
    return eval


def eval_roles(
    instance: InstanceInfo,
    mapping: dict,
    roles: dict[str, Role],
    problem: Problem,
    state: State,
    config: dict,
    action: Optional[Action] = None,
) -> dict[str, RoleDenotation]:
    fstate = _get_dlplan_state(instance, mapping, problem, state, config, action)
    eval = dict()
    for role_string, role in roles.items():
        eval[role_string] = role.evaluate(fstate)
    return eval


def execute_datalog_policy(
    domain: Domain, problem: Problem, datalog_policy: DatalogPolicy, config: ConfigHandler
) -> list[str]:
    log.info(f"Executing policy:\n{datalog_policy}\nin {domain.name} for problem {problem.name}")

    vocabulary = construct_vocabulary_info(domain, config)
    factory = SyntacticElementFactory(vocabulary)
    instance, mapping = construct_instance_info(vocabulary, domain, problem, 0, config)
    object_id_to_name = {o.get_index(): o.get_name() for o in instance.get_objects()}

    concepts = dict()
    roles = dict()
    features = dict()
    log.info("Grounding actions...")
    ground_actions = {(a.name, a.parameters): a for a in ground(domain, problem)}
    log.info("Grounding actions done!")
    for rule in datalog_policy.rules:
        for cond in rule.conds.keys():
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
        log.debug("-" * 80)
        log.info(f"New state: {state_string(state)}")
        found_rule = False

        concepts_eval = eval_concepts(instance, mapping, concepts, problem, state, config)
        roles_eval = eval_roles(instance, mapping, roles, problem, state, config)
        bool_eval = bool_eval_state(instance, mapping, features, problem, state, config)

        for rule in random.sample(list(datalog_policy.rules), len(datalog_policy.rules)):
            log.debug(f"Checking rule: {rule}")
            if not state_satisfies_rule_conds(bool_eval, rule.conds):
                log.debug(f"... Rule conditions not satisfied!")
                continue
            log.debug(f"... Rule conditions satisfied!")
            objects: list[list[str]] = [[] for _ in range(len(rule.parameters))]

            for index, parameter in enumerate(rule.parameters):
                log.debug(f"... Finding valid objects for parameter {parameter}")
                valid_objects = set(list(range(len(instance.get_objects()))))

                for concept in rule.concepts_by_parameter[parameter]:
                    valid_objects &= set(concepts_eval[concept].to_vector())
                    if len(valid_objects) == 0:
                        break

                objects[index] = [object_id_to_name[i] for i in valid_objects]
                random.shuffle(objects[index])
                if len(valid_objects) == 0:
                    break

            log.debug(f"... Found valid objects {objects}")
            if [] in objects:
                log.debug(f"... Rule not applicable! Not all objects found!")
                continue

            log.debug(f"... Checking if rule is applicable")
            action = None
            for object_combination in itertools.product(*objects):
                log.debug(f"... Checking rule with object combination {object_combination}")
                valid = True
                for role_params, rule_roles in rule.roles_by_parameter.items():
                    role_arg_0 = object_combination[rule.parameters.index(role_params[0])]
                    role_arg_1 = object_combination[rule.parameters.index(role_params[1])]
                    for role in rule_roles:
                        if (
                            instance.get_object(role_arg_0).get_index(),
                            instance.get_object(role_arg_1).get_index(),
                        ) not in roles_eval[role].to_vector():
                            log.debug(f"... Role {role} not satisfied for {role_arg_0} and {role_arg_1}")
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                object_combination = tuple(
                    next(c for c in problem.objects | domain.constants if c.name == o) for o in object_combination
                )
                grounded_action = ground_actions.get((rule.name, object_combination))
                if not grounded_action:
                    log.debug(f"... Rule not applicable! No matching action found!")
                    continue
                elif not check_formula(state, grounded_action.precondition):
                    log.debug(f"... Rule not applicable! Precondition not satisfied!")
                    continue
                action = grounded_action
                if not action:
                    continue
                if action:
                    break

            if not action:
                log.debug(f"... Rule not applicable! No matching action found!")
                continue

            if not action:
                log.debug(f"... Rule not applicable! No matching action found!")
                continue

            log.info(f"... Found action {action_string(action)}")
            log.info(f"{rule}")
            log.info(f"Applying action {action_string(action)}")
            found_rule = True
            new_state = get_next_state(apply_action_effects(state, action))
            trace[state] = new_state
            state = new_state
            num_steps += 1
            actions_taken.append(action_string(action))
            break

        if not found_rule:
            log.error(f"No matching rule found for state {state_string(state)}!")
            raise NoActionError(trace, state)

    if not check_formula(state, problem.goal):
        log.error("Goal not reached!")
        raise PolicyExecutionError("Goal not reached!")

    log.info("Goal reached!")
    return actions_taken
