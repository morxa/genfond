import argparse
from genfond.execute_policy import execute_policy
import pddl
import pickle
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', help='domain file')
    parser.add_argument('problem', help='problem file')
    parser.add_argument('policy', help='policy file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    domain = pddl.parse_domain(args.domain)
    problem = pddl.parse_problem(args.problem)
    with open(args.policy, 'rb') as f:
        policy = pickle.load(f)
    actions_taken = execute_policy(domain, problem, policy)
    log.info(f'{len(actions_taken)} actions taken: {", ".join([str(a) for a in actions_taken])}')


if __name__ == '__main__':
    main()
