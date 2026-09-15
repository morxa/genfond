import argparse
import csv
import logging
import os
import pickle
import random
import resource
import signal
import sys
import time

import pddl
import tqdm
from filelock import FileLock
from tqdm.contrib.logging import logging_redirect_tqdm

from genfond.config_handler import DEFAULT_TYPE_CONFIGS, ConfigHandler
from genfond.cost_utils import feature_cost
from genfond.execute_policy import execute_policy

from .iterative_solver import pnames, solve_iteratively
from .shutdown import request_stop, stop_requested

log = logging.getLogger("genfond")


def signal_handler(sig, frame):
    """Request a graceful stop for SIGINT and SIGTERM alike.

    SLURM sends SIGTERM ahead of the hard kill when a job's time limit is reached (see
    genfond.bash's `--signal=B:TERM@600`), and this is also where Ctrl+C (SIGINT) lands. Both
    set the same flag (genfond.shutdown): the iterative solver's round loop stops starting new
    rounds and a clingo solve already in flight is cancelled (Solver.solve polls for exactly
    this), after which main() falls through to the same pickling/verification/stats-writing tail
    as a normal end, just as a --max-wall-time stop does.

    A second signal means the graceful path is not making progress (e.g. stuck outside any
    clingo solve, which is the only place the flag is polled); exit immediately rather than
    leave the job to be SIGKILLed with nothing written.
    """
    if stop_requested():
        log.warning(f"Received {signal.Signals(sig).name} again, exiting immediately ...")
        os._exit(1)
    log.info(f"Received {signal.Signals(sig).name}, requesting a graceful stop ...")
    request_stop()


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
        "--concept-complexity-offset",
        type=int,
        help="cap concept complexity at max(1, complexity - offset) instead of the full per-round complexity",
    )
    config_args.add_argument(
        "--role-complexity-offset",
        type=int,
        help="cap role complexity at max(1, complexity - offset) instead of the full per-round complexity",
    )
    config_args.add_argument(
        "--reset-complexity-on-state-space-change",
        action=argparse.BooleanOptionalAction,
        # Must default to None, not False: ConfigHandler skips None overrides, so a False
        # default would silently overrule the setting from --config on every run.
        default=None,
        help="restart the complexity sweep at min-complexity whenever an example plan or dead end is added",
    )
    config_args.add_argument(
        "--add-problem-after-success",
        action=argparse.BooleanOptionalAction,
        # Must default to None, not False: ConfigHandler skips None overrides, so a False
        # default would silently overrule the setting from --config on every run.
        default=None,
        help="add the next unsolved problem right after a success instead of first climbing feature complexity",
    )
    config_args.add_argument(
        "--final-cost-minimization",
        action=argparse.BooleanOptionalAction,
        # Must default to None, not False: ConfigHandler skips None overrides, so a False
        # default would silently overrule the setting from --config on every run.
        default=None,
        help="with --add-problem-after-success, run the old complexity climb once more on the "
        "final training set once the run has nothing left to add, to look for a cheaper policy",
    )
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
        "--clingo-opt-strategy",
        help='clingo optimisation strategy, e.g. "bb" (default, branch and bound) or "usc" (core-guided)',
    )
    config_args.add_argument(
        "--clingo-option",
        dest="clingo_options",
        action="append",
        # Must default to None, not [], so that ConfigHandler leaves the config value alone.
        default=None,
        metavar="OPT",
        help="extra raw clingo option for the Control; repeatable",
    )
    config_args.add_argument(
        "--solve-time-limit",
        type=float,
        help="wall-clock budget in seconds for a single clingo solve; keeps the best model found so far",
    )
    config_args.add_argument(
        "--max-wall-time",
        type=float,
        help="wall-clock budget in seconds for the whole run (minus wall_time_reserve); stops "
        "starting new rounds and cuts off a running solve once exhausted, then finishes "
        "gracefully (pickles the last policy, verifies it, writes stats) same as a normal end",
    )
    config_args.add_argument(
        "--lazy-pairs",
        action=argparse.BooleanOptionalAction,
        # Must default to None, not False: ConfigHandler skips None overrides.
        default=None,
        help="add the signature separation constraints one batch of violated pairs at a time",
    )
    config_args.add_argument(
        "--lazy-pairs-batch",
        type=int,
        help="how many violated pairs one lazy iteration may add",
    )
    config_args.add_argument(
        "--fix-forced-labels",
        action=argparse.BooleanOptionalAction,
        # Must default to None, not False: ConfigHandler skips None overrides.
        default=None,
        help="pin the good/bad labels that every model agrees on before the search " "(solve_datalog_sig.lp only)",
    )
    config_args.add_argument(
        "--plan-label-heuristic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="bias clingo's decision heuristic towards the transitions on an example plan "
        "(solve_datalog_sig.lp only)",
    )
    config_args.add_argument(
        "--minimize-good-signatures",
        choices=["none", "below", "above"],
        help="minimize the number of good signature classes (solve_datalog_sig.lp only): "
        "'below' only breaks ties after feature cost, 'above' decides fewest good signatures first",
    )
    config_args.add_argument(
        "--minimize-selected-count",
        choices=["none", "below", "above"],
        help="minimize the number of selected concepts/features/roles, instead of just their "
        "total complexity (solve_datalog_sig.lp only): 'below' only breaks ties after feature "
        "cost, 'above' decides fewest selected elements first",
    )
    config_args.add_argument(
        "--seed",
        type=int,
        help="seed the global RNG, which policy execution draws on; needed to compare two runs",
    )
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
    signal.signal(signal.SIGTERM, signal_handler)
    config = ConfigHandler(args.config, args.type, vars(args))
    if args.dump_config:
        with open(args.dump_config, "w") as f:
            f.write(config.dump())
    for component, loglevel in config["log"].items():
        component = f"genfond.{component}"
        log.info(f"Setting log level for {component} to {loglevel}")
        logging.getLogger(component).setLevel(loglevel)
    if config["seed"] is not None:
        # execute_*_policy draws rule order, object bindings and successors from the global RNG,
        # so without this two runs of the identical policy report different solved counts.
        log.info(f'Seeding the global RNG with {config["seed"]}')
        random.seed(config["seed"])
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
    policy, succs, solve_stats = solve_iteratively(domain, problems, config, one_shot=args.one_shot)
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
            # The feature complexity sum; higher-priority levels (e.g. the frontier-transition
            # count, and with minimize_good_signatures/minimize_selected_count="above" the
            # good-signature/selected-element count) are prepended by clingo and only present
            # when their #minimize actually grounds, and with either of them set to "below" the
            # corresponding count is appended after it instead -- see cost_utils.feature_cost.
            "cost": (
                feature_cost(
                    policy.cost,
                    config["minimize_good_signatures"],
                    config.get("minimize_selected_count", "none"),
                )
                if policy
                else 0
            ),
        }
    )

    log.info("Total wall time: {:.2f}s".format(total_wall_time))
    if policy:
        log.info("Best policy solver CPU time: {:.2f}s".format(stats["bestSolveCpuTime"]))
        log.info("Best policy solver wall time: {:.2f}s".format(stats["bestSolveWallTime"]))
    log.info("Total solver CPU time: {:.2f}s".format(stats["totalSolveCpuTime"]))
    log.info("Total CPU time: {:.2f}s".format(total_cpu_time))
    log.info("Total memory usage: {:.2f}MB".format(mem_usage))
    if args.stats:
        lock = FileLock(args.stats + ".lock")
        with lock:
            # The key set differs between runs (e.g. failureReason only exists on failure), so
            # rows appended to an existing file must follow its header or the columns shift.
            fieldnames: list[str] = list(stats.keys())
            file_exists = os.path.isfile(args.stats)
            if file_exists:
                with open(args.stats) as f:
                    fieldnames = next(csv.reader(f))
                dropped = set(stats) - set(fieldnames)
                if dropped:
                    log.warning(f"Stats keys not in the header of {args.stats}, dropped: {sorted(dropped)}")
            with open(args.stats, "a") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(stats)
    if len(succs) == len(problems):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
