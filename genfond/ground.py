import itertools
from pddl.action import Action
from pddl.logic import Predicate, Variable
from pddl.logic.base import UnaryOp, BinaryOp
from pddl.logic.predicates import EqualTo
from pddl.logic.terms import Constant
from pddl.logic.effects import AndEffect, When


def _ground_ops(op, mapping):
    optype = type(op)
    if optype == Constant:
        return op
    if optype == Predicate:
        return Predicate(op.name, *[_ground_ops(t, mapping) for t in op.terms])
    elif optype == Variable:
        try:
            return mapping[op]
        except KeyError as e:
            raise NameError(f'Unknown variable {op}') from e
    elif optype == EqualTo:
        return EqualTo(_ground_ops(op.left, mapping), _ground_ops(op.right, mapping))
    elif issubclass(optype, UnaryOp):
        return optype(_ground_ops(op.argument, mapping))
    elif issubclass(optype, BinaryOp):
        return optype(*[_ground_ops(t, mapping) for t in op.operands])
    elif optype == AndEffect:
        return optype(*[_ground_ops(t, mapping) for t in op.operands])
    elif optype == When:
        return optype(_ground_ops(op.condition, mapping), _ground_ops(op.effect, mapping))
    else:
        raise TypeError("Unknown operator type: {}".format(optype))


def _is_subtype(subtype, supertype, type_dict):
    if subtype == supertype:
        return True
    if subtype in type_dict:
        return _is_subtype(type_dict[subtype], supertype, type_dict)
    return False


def _check_types(constant, variable, type_dict):
    if not type_dict:
        # There are no types, so we assume everything is of the same type.
        return True
    return any(_is_subtype(constant.type_tag, v_type, type_dict) for v_type in variable.type_tags)


def ground_action(domain, problem, action, grounding):
    if not isinstance(action, Action):
        action = [a for a in domain.actions if a.name == action][0]
    mapping = dict(zip(action.parameters, grounding))
    if not all(_check_types(c, v, domain.types) for v, c in mapping.items()):
        return None
    ground_precondition = _ground_ops(action.precondition, mapping)
    ground_effect = _ground_ops(action.effect, mapping)
    return Action(action.name, parameters=grounding, precondition=ground_precondition, effect=ground_effect)


def ground(domain, problem):
    constants = domain.constants | problem.objects
    operators = []
    for action in domain.actions:
        for grounding in itertools.product(constants, repeat=len(action.parameters)):
            op = ground_action(domain, problem, action, grounding)
            if op:
                operators.append(op)
    return operators


def ground_domain_predicates(domain, problem):
    constants = domain.constants | problem.objects
    ground_predicates = set()
    for predicate in domain.predicates:
        for grounding in itertools.product(constants, repeat=predicate.arity):
            mapping = dict(zip(predicate.terms, grounding))
            if not all(_check_types(c, v, domain.types) for v, c in mapping.items()):
                continue
            ground_predicate = _ground_ops(predicate, mapping)
            ground_predicates.add(ground_predicate)
    return ground_predicates
