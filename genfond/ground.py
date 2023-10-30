import itertools
from pddl.action import Action
from pddl.logic import Predicate, Variable
from pddl.logic.base import UnaryOp, BinaryOp


def _ground_ops(op, mapping):
    optype = type(op)
    if optype == Predicate:
        return Predicate(op.name, *[_ground_ops(t, mapping) for t in op.terms])
    elif optype == Variable:
        if op in mapping:
            return mapping[op]
        else:
            return mapping[op]
    elif issubclass(optype, UnaryOp):
        return optype(_ground_ops(op.argument, mapping))
    elif issubclass(optype, BinaryOp):
        return optype(*[_ground_ops(t, mapping) for t in op.operands])
    else:
        raise ValueError("Unknown operator type: {}".format(optype))


def ground(domain, problem):
    constants = domain.constants | problem.objects
    operators = []
    for action in domain.actions:
        for grounding in itertools.product(constants,
                                           repeat=len(action.parameters)):
            mapping = dict(zip(action.parameters, grounding))
            parameters = grounding
            ground_precondition = _ground_ops(action.precondition, mapping)
            ground_effect = _ground_ops(action.effect, mapping)
            op = Action(action.name,
                        parameters=parameters,
                        precondition=ground_precondition,
                        effect=ground_effect)
            operators.append(op)
    return operators
