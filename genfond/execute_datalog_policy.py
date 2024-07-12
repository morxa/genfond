import itertools
import logging
import random
from dlplan.core import SyntacticElementFactory
from pddl.logic.terms import Constant
from .execute_rule_policy import eval_state, bool_eval_state, state_satisfies_rule_conds
from .feature_generator import construct_vocabulary_info, construct_instance_info, _get_state_from_goal
from .ground import ground_action
from .state_space_generator import check_formula, apply_action_effects

log = logging.getLogger(__name__)


def get_next_state(states):
    return random.choice([state for state in states])


def action_string(action) -> str:
    return f'{action.name}({",".join([str(p) for p in action.parameters])})'


def state_string(state) -> str:
    return ",".join([str(p) for p in state])


def execute_datalog_policy(domain, problem, datalog_policy, config):
    log.info(f'Executing policy:\n{datalog_policy}\nin {domain.name} for problem {problem.name}')

    vocabulary = construct_vocabulary_info(domain)
    factory = SyntacticElementFactory(vocabulary)
    instance, mapping = construct_instance_info(vocabulary, domain, problem, 0)
    object_id_to_name = {o.get_index(): o.get_name() for o in instance.get_objects()}

    concepts = dict()
    roles = dict()
    features = dict()
    for rule in datalog_policy.rules:
        for cond, _ in rule.conds.items():
            if cond.startswith('b_'):
                features[cond] = factory.parse_boolean(cond)
            elif cond.startswith('n_'):
                features[cond] = factory.parse_numerical(cond)
            else:
                raise ValueError(f'Unknown feature type: {cond}')
        for rule_concepts in rule.concepts_by_parameter.values():
            for concept in rule_concepts:
                concepts[concept] = factory.parse_concept(concept)
        for rule_roles in rule.roles_by_parameter.values():
            for role in rule_roles:
                roles[role] = factory.parse_role(role)

    state = problem.init
    goal_state = _get_state_from_goal(problem.goal)
    num_steps = 0
    actions_taken = []
    max_steps = config['policy_steps']
    while not check_formula(state, problem.goal) and (max_steps <= 0 or num_steps < max_steps):
        log.info(f'New state: {state_string(state)}')
        found_rule = False

        eval = eval_state(instance, mapping, concepts | roles, state, goal_state)
        bool_eval = bool_eval_state(instance, mapping, features, state, goal_state)

        for rule in datalog_policy.rules:
            log.debug(f'Checking rule {rule}')
            if not state_satisfies_rule_conds(bool_eval, rule.conds):
                log.debug(f'... Rule conditions not satisfied!')
                continue
            log.debug(f'... Rule conditions satisfied!')
            objects = [[] for _ in range(len(rule.parameters))]

            for index, parameter in enumerate(rule.parameters):
                log.debug(f'... Finding valid objects for parameter {parameter}')
                valid_objects = set(list(range(len(instance.get_objects()))))

                for concept in rule.concepts_by_parameter[parameter]:
                    valid_objects &= set(eval[concept].to_vector())
                    if len(valid_objects) == 0:
                        break

                objects[index] = [object_id_to_name[i] for i in valid_objects]
                if len(valid_objects) == 0:
                    break

            log.debug(f'... Found valid objects {objects}')
            if [] in objects:
                log.debug(f'... Rule not applicable! Not all objects found!')
                continue

            log.debug(f'... Checking if rule is applicable')
            action = None
            for object_combination in itertools.product(*objects):
                log.debug(f'... Checking rule with object combination {object_combination}')
                valid = True
                for role_params, rule_roles in rule.roles_by_parameter.items():
                    role_arg_0 = object_combination[rule.parameters.index(role_params[0])]
                    role_arg_1 = object_combination[rule.parameters.index(role_params[1])]
                    for role in rule_roles:
                        if (instance.get_object(role_arg_0).get_index(),
                                instance.get_object(role_arg_1).get_index()) not in eval[role].to_vector():
                            log.debug(f'... Role {role} not satisfied for {role_arg_0} and {role_arg_1}')
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                object_combination = tuple(
                    next(c for c in problem.objects | domain.constants if c.name == o) for o in object_combination)
                grounded_action = ground_action(domain, problem, rule.name, object_combination)
                if grounded_action and check_formula(state, grounded_action.precondition):
                    action = grounded_action
                    break

            if not action:
                log.debug(f'... Rule not applicable! No matching action found!')
                continue

            log.info(f'... Found matching action {action_string(action)}! Applying rule!')
            found_rule = True
            state = get_next_state(apply_action_effects(state, action))
            num_steps += 1
            actions_taken.append(action_string(action))
            break

        if not found_rule:
            log.error(f'No matching rule found for state {state_string(state)}!')
            raise RuntimeError('No matching rule found!')

    if not check_formula(state, problem.goal):
        log.error('Goal not reached!')
        raise RuntimeError('Goal not reached!')

    log.info('Goal reached!')
    return actions_taken
