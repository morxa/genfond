from typing import Any, Optional

from .datalog_policy import DatalogPolicy
from .generate_datalog_policy import generate_datalog_policy
from .generate_rule_policy import generate_rule_policy
from .policy import PolicyType
from .rule_policy import Policy


def generate_policy(
    solution: dict[str, Any], policy_type: PolicyType = PolicyType.EXACT, save_file: Optional[str] = None
) -> Policy | DatalogPolicy:
    if policy_type == PolicyType.DATALOG:
        return generate_datalog_policy(solution)
    return generate_rule_policy(solution, policy_type, save_file=save_file)
