from .generate_datalog_policy import generate_datalog_policy
from .generate_rule_policy import generate_rule_policy
from .policy import PolicyType


def generate_policy(solution, policy_type=PolicyType.EXACT):
    if policy_type == PolicyType.DATALOG:
        return generate_datalog_policy(solution)
    return generate_rule_policy(solution, policy_type)
