import argparse
import logging
import sys

import pddl
import tqdm

from genfond.state_space_generator import StateSpaceGraph

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Check if the given problems are solvable")
    parser.add_argument("domain_file")
    parser.add_argument("problem_file", nargs="*")
    args = parser.parse_args()
    domain = pddl.parse_domain(args.domain_file)
    log.info("Parsing problems ...")
    problems = []
    for f in tqdm.tqdm(args.problem_file, disable=None):
        problems.append(pddl.parse_problem(f))
    problems.sort(key=lambda p: len(p.objects))
    solvable = True
    for i, problem in enumerate(problems):
        try:
            log.info(f"Checking {i}/{len(problems)} problem {problem.name} ...")
            _ = StateSpaceGraph(domain, problem)
        except AssertionError:
            log.warning("Problem {} is not solvable".format(problem.name))
            solvable = False
    sys.exit(0 if solvable else 1)


if __name__ == "__main__":
    main()
