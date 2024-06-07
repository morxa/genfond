import itertools
import logging
import random
from dlplan.core import SyntacticElementFactory
from pddl.logic.terms import Constant
from .execute_policy import eval_state
from .feature_generator import construct_vocabulary_info, construct_instance_info, _get_state_from_goal
from .ground import ground
from .state_space_generator import check_formula, apply_action_effects

log = logging.getLogger(__name__)


def get_next_state(states):
    return random.choice([state for state in states])


def action_string(action) -> str:
    return f'{action.name}({",".join([str(p) for p in action.parameters])})'


def state_string(state) -> str:
    return ",".join([str(p) for p in state])


def execute_datalog_policy(domain, problem, datalog_policy, max_steps=0):
    log.info(f'Executing policy:\n{datalog_policy}\nin {domain.name} for problem {problem.name}')

    log.debug("Grounding actions...")
    grounded_actions = ground(domain, problem)
    log.debug("Grounding actions done.")
    
    vocabulary = construct_vocabulary_info(domain)
    factory = SyntacticElementFactory(vocabulary)
    instance, mapping = construct_instance_info(vocabulary, domain, problem, 0)
    object_id_to_name = {o.get_index(): o.get_name() for o in instance.get_objects()}
    
    features = {}
    for rule in datalog_policy.rules:
        for concepts in rule.tail_by_parameter.values():
            for concept in concepts:
                features[concept] = factory.parse_concept(concept)

    state = problem.init
    goal_state = _get_state_from_goal(problem.goal)
    num_steps = 0
    actions_taken = []
    while not check_formula(state, problem.goal) and (max_steps <= 0 or num_steps < max_steps):
        log.info(f'New state: {state_string(state)}')
        found_rule = False
        
        eval = eval_state(instance, mapping, features, state, goal_state)
        
        for rule in datalog_policy.rules:
            log.debug(f'Checking rule {rule}')
            
            objects = [[] for _ in range(len(rule.parameters))]
            
            for index, parameter in enumerate(rule.parameters):
                log.debug(f'... Finding valid objects for parameter {parameter}')
                valid_objects = set(list(range(len(instance.get_objects()))))
                
                for concept in rule.tail_by_parameter[parameter]:
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
                object_combination = tuple(Constant(o) for o in object_combination)
                for grounded_action in grounded_actions:
                    if grounded_action.name == rule.name and grounded_action.parameters == object_combination and check_formula(state, grounded_action.precondition):
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
