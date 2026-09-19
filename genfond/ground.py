import itertools
from typing import Optional

from pddl.action import Action
from pddl.core import Domain, Problem
from pddl.custom_types import name as name_type
from pddl.logic import Predicate, Variable
from pddl.logic.base import And, BinaryOp, Formula, UnaryOp
from pddl.logic.effects import When
from pddl.logic.functions import BinaryFunction
from pddl.logic.functions import EqualTo as FunctionEqualTo
from pddl.logic.functions import NumericFunction, NumericValue
from pddl.logic.predicates import EqualTo
from pddl.logic.terms import Constant, Term

type TypeTag = Optional[name_type]


def _ground_term(term: Term, mapping: dict[Variable, Constant]) -> Constant:
    term_type = type(term)
    if isinstance(term, Constant):
        return term
    elif isinstance(term, Variable):
        try:
            return mapping[term]
        except KeyError as e:
            raise NameError(
                f'Unknown variable {term}, known variables: {", ".join([str(k) for k in mapping.keys()])}'
            ) from e
    else:
        raise TypeError(f"{term}: unknown term type: {term_type}")


def _ground_formula(op: Formula, mapping: dict[Variable, Constant]) -> Formula:
    optype = type(op)
    if optype == Predicate:
        return Predicate(op.name, *[_ground_term(t, mapping) for t in op.terms])
    elif optype == EqualTo:
        return EqualTo(_ground_term(op.left, mapping), _ground_term(op.right, mapping))
    elif issubclass(optype, UnaryOp):
        return optype(_ground_formula(op.argument, mapping))
    elif issubclass(optype, BinaryOp):
        return optype(*[_ground_formula(t, mapping) for t in op.operands])
    elif optype == And:
        return optype(*[_ground_formula(t, mapping) for t in op.operands])
    elif optype == When:
        return optype(_ground_formula(op.condition, mapping), _ground_formula(op.effect, mapping))
    elif issubclass(optype, BinaryFunction):
        return optype(*[_ground_formula(t, mapping) for t in op.operands])
    elif optype == NumericValue:
        return op
    elif issubclass(optype, NumericFunction):
        return optype(op.name, *[_ground_term(t, mapping) for t in op.terms])
    else:
        raise TypeError(f"{op}: unknown operator type: {optype}")


def _is_subtype(subtype: TypeTag, supertype: TypeTag, type_dict: dict[TypeTag, TypeTag]) -> bool:
    if subtype == supertype:
        return True
    if subtype in type_dict:
        return _is_subtype(type_dict[subtype], supertype, type_dict)
    return False


def _check_types(constant: Constant, variable: Variable, type_dict: dict[TypeTag, TypeTag]) -> bool:
    if not type_dict:
        # There are no types, so we assume everything is of the same type.
        return True
    return any(_is_subtype(constant.type_tag, v_type, type_dict) for v_type in variable.type_tags)


def _stable_constants(domain: Domain, problem: Problem) -> list[Constant]:
    """The domain constants and problem objects in a process-independent order.

    `domain.constants | problem.objects` is a frozenset of `pddl` `Constant`s, and
    `Constant.__hash__` is `hash((Constant, name))` -- it mixes in the hash of the *class
    object*, which is the default identity hash and therefore an address. Two runs of the same
    code with the same `PYTHONHASHSEED` iterate such a set in different orders, because ASLR
    moves the class. `Predicate.__hash__` hashes its terms, so the same is true of a `State`
    (`frozenset[Predicate]`) and of `domain.predicates` (whose terms are `Variable`s, hashed the
    same way). Sorting by name removes the dependence; `feature_generator.py` does the same for
    the orders DLPlan and the ASP instance inherit.
    """
    return sorted(domain.constants | problem.objects, key=lambda c: str(c.name))


def ground_action(domain: Domain, action, grounding: tuple[Constant]) -> Optional[Action]:
    if not isinstance(action, Action):
        action = [a for a in domain.actions if a.name == action][0]
    mapping = dict(zip(action.parameters, grounding))
    if not all(_check_types(c, v, domain.types) for v, c in mapping.items()):
        return None
    ground_precondition = _ground_formula(action.precondition, mapping)
    ground_effect = _ground_formula(action.effect, mapping)
    return Action(
        action.name,
        parameters=grounding,
        precondition=ground_precondition,
        effect=ground_effect,
    )


def _typed_candidates(domain: Domain, constants: list[Constant], terms) -> list[list[Constant]]:
    """The constants admissible at each of `terms`, each in the order of `constants`.

    `itertools.product(constants, repeat=arity)` enumerates |constants|^arity tuples and discards
    every one that is not well typed, which for a 4-ary action over a few dozen objects means
    hundreds of thousands of tuples to produce a few hundred groundings. Filtering per parameter
    first enumerates exactly the well-typed tuples instead. The result is the same set of
    groundings, in the same relative order, and both callers sort or set-collect it anyway.
    """
    return [[c for c in constants if _check_types(c, term, domain.types)] for term in terms]


def ground(domain: Domain, problem: Problem) -> list[Action]:
    constants = _stable_constants(domain, problem)
    operators = []
    for action in domain.actions:
        for grounding in itertools.product(*_typed_candidates(domain, constants, action.parameters)):
            op = ground_action(domain, action, grounding)
            if op:
                operators.append(op)
    return sorted(operators, key=lambda a: (a.name, a.parameters))


def ground_domain_predicates(domain: Domain, problem: Problem) -> set[Predicate]:
    constants = _stable_constants(domain, problem)
    ground_predicates = set()
    for predicate in domain.predicates:
        for grounding in itertools.product(*_typed_candidates(domain, constants, predicate.terms)):
            mapping = dict(zip(predicate.terms, grounding))
            ground_predicate = _ground_formula(predicate, mapping)
            ground_predicates.add(ground_predicate)
    return ground_predicates


def action_string(action: Action) -> str:
    return f'{action.name}({",".join([str(p) for p in action.parameters])})'


def state_string(state) -> str:
    state_str = []
    for p in state:
        if isinstance(p, Predicate):
            state_str.append(f'{p.name}({",".join([str(p) for p in p.terms])})')
        elif isinstance(p, FunctionEqualTo):
            state_str.append(f"{p.operands[0]}={p.operands[1]}")
        else:
            raise ValueError("Unknown state type: {}".format(type(p)))
    return ",".join(sorted(state_str))
