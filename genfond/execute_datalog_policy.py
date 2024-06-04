from .feature_generator import FeaturePool
from .ground import ground
from .state_space_generator import check_formula, apply_action_effects
import itertools
import logging
import random

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
    
    features = []
    for rule in datalog_policy.rules:
        for concepts in rule.tail_by_parameter.values():
            features.extend(concepts)
    
    featurePool = FeaturePool(domain, [problem], max_complexity=15, preset_features=features)

    state = problem.init
    num_steps = 0
    actions_taken = []
    while not check_formula(state, problem.goal) and (max_steps <= 0 or num_steps < max_steps):
        log.info(f'New state: {state_string(state)}')
        found_rule = False
        
        for rule in datalog_policy.rules:
            log.debug(f'Checking rule {rule}')
            
            objects = [[] for _ in range(len(rule.parameters))]
            
            for index, parameter in enumerate(rule.parameters):
                log.debug(f'Checking rule {rule} for parameter {parameter}')
                
                for object in domain.constants | problem.objects:
                    success = True
                    for concept in rule.tail_by_parameter[parameter]:
                        log.debug(f'Checking rule {rule} for parameter {parameter}, checking concept {concept} for object {object}')
                        if not featurePool.evaluate_concept_for_object(concept, state, str(object)):
                            success = False
                            break
                    if success:
                        objects[index].append(object)
                        log.debug(f'Checking rule {rule} for parameter {parameter}, found matching object {object}')
                        
                if len(objects[index]) == 0:
                    break
                
            if [] in objects:
                log.debug(f'Checking rule {rule}, not all objects found!')
                continue
            
            action = None
            for object_combination in itertools.product(*objects):
                object_combination = tuple(object_combination)
                log.debug(f'Checking rule {rule} for object combination {object_combination}')
                for grounded_action in grounded_actions:
                    if grounded_action.name == rule.name and grounded_action.parameters == object_combination and check_formula(state, grounded_action.precondition):
                        action = grounded_action
                        break
        
            if not action:
                log.debug(f'Checking rule {rule} no matching action found!')
                continue
                
            log.info(f'Found matching rule {rule} with action {action_string(action)}!')
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
