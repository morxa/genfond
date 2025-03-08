import argparse
import logging
import pickle

import pddl

from genfond.config_handler import ConfigHandler
from genfond.datalog_policy import DatalogPolicy
from genfond.execute_policy import execute_policy

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", help="domain file")
    parser.add_argument("problem", help="problem file")
    parser.add_argument("policy", help="policy file")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--config", type=argparse.FileType("r"), help="config file")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    with open(args.policy, "rb") as f:
        policy = pickle.load(f)
    type = "datalog" if isinstance(policy, DatalogPolicy) else "rule"
    config = ConfigHandler(args.config, type, vars(args))
    domain = pddl.parse_domain(args.domain)
    problem = pddl.parse_problem(args.problem)
    actions_taken = execute_policy(domain, problem, policy, config)
    log.info(f'{len(actions_taken)} actions taken: {", ".join([str(a) for a in actions_taken])}')


if __name__ == "__main__":
    main()
