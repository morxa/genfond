from .config_handler import ConfigHandler
from .datalog_policy import DatalogPolicy
from .execute_datalog_policy import execute_datalog_policy
from .execute_rule_policy import execute_rule_policy


def execute_policy(domain, problem, policy, config=None):
    if isinstance(policy, DatalogPolicy):
        if not config:
            config = ConfigHandler(type="datalog")
        return execute_datalog_policy(domain, problem, policy, config)
    else:
        if not config:
            config = ConfigHandler()
        return execute_rule_policy(domain, problem, policy, config)
