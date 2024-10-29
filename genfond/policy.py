from enum import Enum


class PolicyType(Enum):
    EXACT = 1
    CONSTRAINED = 2
    DATALOG = 3
