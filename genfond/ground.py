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


def ground(domain, problem):
    constants = domain.constants | problem.objects
    operators = []
    for action in domain.actions:
        for grounding in itertools.product(constants, repeat=len(action.parameters)):
            mapping = dict(zip(action.parameters, grounding))
            parameters = grounding
            ground_precondition = _ground_ops(action.precondition, mapping)
            ground_effect = _ground_ops(action.effect, mapping)
            op = Action(action.name, parameters=parameters, precondition=ground_precondition, effect=ground_effect)
            operators.append(op)
    return operators


def ground_domain_predicates(domain, problem):
    constants = domain.constants | problem.objects
    ground_predicates = set()
    for predicate in domain.predicates:
        for grounding in itertools.product(constants, repeat=predicate.arity):
            mapping = dict(zip(predicate.terms, grounding))
            ground_predicate = _ground_ops(predicate, mapping)
            ground_predicates.add(ground_predicate)
    return ground_predicates
