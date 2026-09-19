"""Execute a pickled policy on a set of problems and count how many it solves.

    python scripts/eval_policy.py --seed 0 domain.pddl out.policy p*.pddl

A problem counts as solved only if every one of --policy-iterations executions reaches the
goal, mirroring the verification loop in genfond/__main__.py. Used for the held-out
generalisation check (domains/suites/blocks3ops-heldout) that the training run never sees.
"""

import argparse
import logging
import pickle
import random
import sys
import time
from pathlib import Path

# The package is not installed into the venv; import it from the checkout this script lives in.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pddl  # noqa: E402

from genfond.config_handler import ConfigHandler  # noqa: E402
from genfond.datalog_policy import DatalogPolicy  # noqa: E402
from genfond.execute_policy import execute_policy  # noqa: E402

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", help="domain file")
    parser.add_argument("policy", help="pickled policy")
    parser.add_argument("problems", nargs="+", help="problem files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-i", "--policy-iterations", type=int, default=None)
    parser.add_argument("--policy-steps", type=int, default=None)
    parser.add_argument("--config", type=argparse.FileType("r"))
    args = parser.parse_args()
    with open(args.policy, "rb") as f:
        policy = pickle.load(f)
    ptype = "datalog" if isinstance(policy, DatalogPolicy) else "state"
    config = ConfigHandler(args.config, ptype, vars(args))
    random.seed(args.seed)
    domain = pddl.parse_domain(args.domain)
    solved = []
    for pf in args.problems:
        problem = pddl.parse_problem(pf)
        start = time.perf_counter()
        try:
            for _ in range(config["policy_iterations"]):
                execute_policy(domain, problem, policy, config)
            solved.append(problem.name)
            status = "solved"
        except RuntimeError as e:
            status = f"FAILED ({e})"
        log.info(f"{problem.name}: {status} [{time.perf_counter() - start:.1f}s]")
    log.info(f"Policy solves {len(solved)} out of {len(args.problems)} problems")


if __name__ == "__main__":
    main()
