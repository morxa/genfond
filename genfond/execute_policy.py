from typing import Optional

from pddl.action import Action

from .config_handler import ConfigHandler
from .datalog_policy import DatalogPolicy
from .execute_datalog_policy import execute_datalog_policy
from .execute_rule_policy import PolicyExecutionError, execute_rule_policy


def execute_policy(
    domain,
    problem,
    policy,
    config=None,
    time_limit: Optional[float] = None,
    out_actions: Optional[list[Action]] = None,
):
    """Execute `policy` on `problem` and return the list of action strings it applied.

    `out_actions`, when given, is filled with the ground actions of that same trajectory (see
    the executors' own docstrings). Callers that do not need the trajectory pass nothing and
    are unaffected.
    """
    if not policy:
        raise PolicyExecutionError("Empty policy")
    if isinstance(policy, DatalogPolicy):
        if not config:
            config = ConfigHandler(type="datalog")
        return execute_datalog_policy(domain, problem, policy, config, time_limit=time_limit, out_actions=out_actions)
    else:
        if not config:
            config = ConfigHandler()
        return execute_rule_policy(domain, problem, policy, config, time_limit=time_limit, out_actions=out_actions)
