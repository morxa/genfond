from .config_handler import ConfigHandler
from .datalog_policy import DatalogPolicy
from .execute_datalog_policy import execute_datalog_policy
from .execute_rule_policy_dlplan import execute_rule_policy_dl
from .execute_rule_policy_wlplan import execute_rule_policy_wl


def execute_policy_dl(domain, problem, policy, config=None):
    if isinstance(policy, DatalogPolicy):
        if not config:
            config = ConfigHandler(type="datalog")
        return execute_datalog_policy(domain, problem, policy, config)
    else:
        if not config:
            config = ConfigHandler()
        return execute_rule_policy_dl(domain, problem, policy, config)


def execute_policy_wl(domain, problem, policy, config=None):
    if not config:
        config = ConfigHandler()
    return execute_rule_policy_wl(domain, problem, policy, config)
