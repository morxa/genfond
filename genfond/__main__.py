import argparse
import csv
import logging
import os
import pickle
import resource
import signal
import sys
import time

import pddl
import tqdm
from filelock import FileLock
from tqdm.contrib.logging import logging_redirect_tqdm

from genfond.config_handler import DEFAULT_TYPE_CONFIGS, ConfigHandler
from genfond.execute_policy import execute_policy

from .iterative_solver import pnames, solve, solve_iteratively
from .problem_iterator import MAX_COST

log = logging.getLogger("genfond")


def signal_handler(sig, frame):
    log.info(f"Received {signal.Signals(sig).name}, exiting ...")
    os.kill(os.getpid(), signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_file")
    parser.add_argument("problem_file", nargs="*")
    parser.add_argument("--one-shot", action="store_true", help="solve all problems at once")
    parser.add_argument("--name", help="Name of the problem set (default: domain name)")
    parser.add_argument("--output", "-o", help="Output file for the resulting policy (as pickle dump)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--stats", help="file to dump stats to")
    parser.add_argument("--config", type=argparse.FileType("r"), help="config file for parameters")
    parser.add_argument("--dump-config", help="dump effective config to file")
    parser.add_argument("--dump-clingo-program", help="dump clingo program to file")
    parser.add_argument(
        "--type",
        choices=DEFAULT_TYPE_CONFIGS.keys(),
        default="state",
        help="generate policies of the given type",
    )
    config_args = parser.add_argument_group("config", "Overwrite config parameters")
    config_args.add_argument(
        "--min-complexity",
        type=int,
        help="start policy search with this max complexity",
    )
    config_args.add_argument("--max-complexity", type=int, help="stop policy search with this max complexity")
    config_args.add_argument(
        "-i",
        "--policy-iterations",
        type=int,
        help="number of policy iterations for testing",
    )
    config_args.add_argument(
        "--policy-steps",
        type=int,
        help="number of steps to execute policy for testing (0 for no limit)",
    )
    config_args.add_argument(
        "-n",
        "--num-threads",
        type=int,
        help='number of threads to use; "None" uses all available threads',
    )
    config_args.add_argument("--max-memory", type=int, help="maximum memory to use in MB")
    config_args.add_argument(
        "--dump-failed-policies",
        action="store_true",
        help="dump failed policies to file",
    )
    config_args.add_argument(
        "--keep-going",
        action="store_true",
        help="keep going after one training problem failed",
    )
    config_args.add_argument(
        "--continue-after-error",
        action="store_true",
        help="continue after error in policy execution",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    signal.signal(signal.SIGINT, signal_handler)
    config = ConfigHandler(args.config, args.type, vars(args))
    if args.dump_config:
        with open(args.dump_config, "w") as f:
            f.write(config.dump())
    for component, loglevel in config["log"].items():
        component = f"genfond.{component}"
        log.info(f"Setting log level for {component} to {loglevel}")
        logging.getLogger(component).setLevel(loglevel)
    if config["max_memory"]:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (config["max_memory"] * 1024 * 1024, hard))
    total_wall_time_start = time.perf_counter()
    total_cpu_time_start = time.process_time()
    log.info("Parsing domain ...")
    domain = pddl.parse_domain(args.domain_file)
    log.info("Parsing problems ...")
    problems = []
    for f in tqdm.tqdm(args.problem_file, disable=None):
        problems.append(pddl.parse_problem(f))
    name = args.name if args.name else domain.name
    log.info("Starting policy generation for domain {}".format(name))
    log.info(f"Generating policies of type {args.type}")
    stats = {
        "domain": name,
        "constraintType": args.type,
    }
    if args.one_shot:
        solve_cpu_time_start = time.process_time()
        solution = solve(
            domain,
            problems,
            config=config,
            complexity=config["max_complexity"],
            all_generators=False,
            max_cost=MAX_COST,
            max_prune_cost=MAX_COST,
        )
        solve_cpu_time = time.process_time() - solve_cpu_time_start
        log.info(f"CPU time: {solve_cpu_time:.2f}s")
        mem_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024
        log.info("Memory usage: {:.2f}MB".format(mem_usage))
        if solution:
            policy, solve_stats = solution
            stats.update(solve_stats)
        else:
            log.error("No policy found")
            sys.exit(1)
        log.info(f"Policy:\n{policy}")
        if args.output:
            with open(args.output, "wb") as f:
                pickle.dump(policy, f)
        sys.exit(0)
    policy, succs, solve_stats = solve_iteratively(domain, problems, config)
    stats.update(solve_stats)
    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump(policy, f)
    log.info("Verifying policy ...")
    with logging_redirect_tqdm():
        for problem in tqdm.tqdm([p for p in problems if p not in succs], disable=None):
            try:
                for _ in tqdm.trange(config["policy_iterations"], leave=False, disable=None):
                    execute_policy(domain, problem, policy, config)
                succs.append(problem)
            except RuntimeError:
                log.error("Policy does not solve {}".format(problem.name))
    log.info(
        "Policy solves {} out of {} problems, unsolved: {}".format(
            len(succs), len(problems), pnames([p for p in problems if p not in succs])
        )
    )
    log.info("Final policy: {}".format(policy))
    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump(policy, f)
    total_wall_time = time.perf_counter() - total_wall_time_start
    total_cpu_time = time.process_time() - total_cpu_time_start
    mem_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024
    stats.update(
        {
            "totalWallTime": total_wall_time,
            "totalCpuTime": total_cpu_time,
            "problems": len(problems),
            "solved": len(succs),
            "maxProblemSize": max(len(p.objects) for p in problems) if succs else 0,
            "memUsage": mem_usage,
            #'numFeatures': len(policy.features),
            #'numConstraints': max(len(policy.state_constraints), len(policy.constraints)),
            "cost": policy.cost[0] if policy else 0,
        }
    )

    log.info("Total wall time: {:.2f}s".format(total_wall_time))
    log.info("Best policy solver CPU time: {:.2f}s".format(stats["bestSolveCpuTime"]))
    log.info("Best policy solver wall time: {:.2f}s".format(stats["bestSolveWallTime"]))
    log.info("Total solver CPU time: {:.2f}s".format(stats["totalSolveCpuTime"]))
    log.info("Total CPU time: {:.2f}s".format(total_cpu_time))
    log.info("Total memory usage: {:.2f}MB".format(mem_usage))
    if args.stats:
        lock = FileLock(args.stats + ".lock")
        with lock:
            file_exists = os.path.isfile(args.stats)
            with open(args.stats, "a") as f:
                writer = csv.DictWriter(f, fieldnames=stats.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(stats)
    if len(succs) == len(problems):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
