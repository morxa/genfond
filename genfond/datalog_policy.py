import logging
import re
from typing import Any, Collection, Optional

from frozendict import frozendict

from .rule_policy import Cond, cond_to_str

log = logging.getLogger("genfond.policy.datalog")

ACTION_REGEX = r"^([^(]+)\(([^(]*)\)$"
RULE_VARS = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def split_action_string(action: str) -> tuple[str, list[str]]:
    match = re.search(ACTION_REGEX, action.strip('"'))
    if not match:
        raise ValueError(f"Invalid action string: {action}")
    name = match.groups()[0]
    if match.groups()[1]:
        parameters = match.groups()[1].replace(" ", "").split(",")
    else:
        parameters = []
    return name, parameters


class DatalogPolicyRule:

    def __init__(
        self,
        head: str,
        concepts: Optional[list[tuple[str, str]]] = None,
        roles: Optional[list[tuple[str, str, str]]] = None,
        conds: Optional[dict[str, Cond]] = None,
    ):
        if not concepts:
            concepts = []
        if not roles:
            roles = []
        if not conds:
            conds = dict()
        self.name, self.parameters = split_action_string(head)
        concepts_by_parameter: dict[str, list[str]] = {name: [] for name in self.parameters}
        roles_by_parameter: dict[tuple[str, str], list[str]] = {}
        for parameter, concept in concepts or []:
            if parameter not in self.parameters:
                raise ValueError(f"Free variable {parameter} not in head parameters {self.parameters}!")
            concepts_by_parameter[parameter].append(concept.replace(" ", ""))
        for param1, param2, role in roles:
            if param1 not in self.parameters:
                raise ValueError(f"Free variable {param1} not in head parameters {self.parameters}!")
            if param2 not in self.parameters:
                raise ValueError(f"Free variable {param2} not in head parameters {self.parameters}!")
            roles_by_parameter.setdefault((param1, param2), []).append(role)
        self.concepts_by_parameter = {name: frozenset(concepts) for name, concepts in concepts_by_parameter.items()}
        self.roles_by_parameter = {params: frozenset(roles) for params, roles in roles_by_parameter.items()}
        self.conds = frozendict(conds)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DatalogPolicyRule):
            raise NotImplementedError
        if self.name != other.name:
            return False
        if len(self.parameters) != len(other.parameters):
            return False
        if self.conds != other.conds:
            return False
        for p_self, p_other in zip(self.parameters, other.parameters):
            if self.concepts_by_parameter[p_self] != other.concepts_by_parameter[p_other]:
                return False
        if self.roles_by_parameter.keys() != other.roles_by_parameter.keys():
            return False
        for params in self.roles_by_parameter.keys():
            if self.roles_by_parameter[params] != other.roles_by_parameter[params]:
                return False
        return True

    def __repr__(self) -> str:
        state_cond_strs = [cond_to_str(cond, val) for cond, val in self.conds.items()]
        state_cond_strs.sort()
        concept_cond_strs = []
        roles_cond_strs = []
        for parameter, concepts in self.concepts_by_parameter.items():
            concept_cond_strs.extend([f"{parameter} ∊ {concept}" for concept in concepts])
        concept_cond_strs.sort()
        for parameters, roles in self.roles_by_parameter.items():
            roles_cond_strs.extend([f"{parameters[0]} {role} {parameters[1]}" for role in roles])
        roles_cond_strs.sort()
        conds = state_cond_strs + concept_cond_strs + roles_cond_strs
        return f'{self.name}({', '.join(self.parameters)}){f" :- {', '.join(conds)}" if conds else ""}.'

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                len(self.parameters),
                self.conds,
                *tuple(self.concepts_by_parameter[p] for p in self.parameters),
                *tuple((params, roles) for params, roles in self.roles_by_parameter.items()),
            )
        )


class DatalogPolicy:

    def __init__(self, rules: Collection[DatalogPolicyRule], cost: Optional[tuple[int]] = None):
        self.cost = cost
        self.rules = frozenset(rules)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DatalogPolicy):
            raise NotImplementedError
        return self.rules == other.rules

    def __repr__(self) -> str:
        return f"{len(self.rules)} rules\n" + "\n".join([str(r) for r in self.rules])

    def __hash__(self) -> int:
        return hash(self.rules)
