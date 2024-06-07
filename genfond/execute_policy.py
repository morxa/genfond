from .datalog_policy import DatalogPolicy
from .execute_rule_policy import execute_rule_policy
from .execute_datalog_policy import execute_datalog_policy


def execute_policy(domain, problem, policy):
    if isinstance(policy, DatalogPolicy):
        return execute_datalog_policy(domain, problem, policy)
    else:
        return execute_rule_policy(domain, problem, policy)
