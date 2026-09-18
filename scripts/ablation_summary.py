"""Summarise a results-* directory produced by scripts/ablation.sh.

    python scripts/ablation_summary.py results-...-abl

Prints two markdown tables (domains x arms): solved/problems per seed with the mean, and median
wall time per arm. Also lists (domain, arm, seed) triples with no result, so gaps -- a job still
running, killed before it logged anything, or never submitted -- are visible.

Why this does not key off stats.csv (unlike scripts/ab_summary.py): every ablation job shares the
same --type (datalog-sig) and is named by --name <domainname> alone (see
scripts/ablation_job.bash), so stats.csv rows for the same domain are indistinguishable between
arms and seeds -- there is no config/policy-path column to key on (checked against
genfond/__main__.py and genfond/checkpoint.py: the stats dict never carries the output or config
path). What *is* unique per job is its SLURM array output file, named "<TAG>-<arm>-<A>_<a>.out"
by ablation.sh's `-o`, where `<a>` is the array task id -- which is exactly the 1-indexed line
number of that job in results-dir/manifest-<arm>.txt. This script uses that mapping (manifest
line <-> out file) to identify every job, then parses its own log for the same two lines
scripts/eval_policy.py-style runs already rely on: "Policy solves X out of Y problems" and
"Total wall time: Ns" (both logged by genfond/__main__.py).
"""

import argparse
import glob
import os
import re
import statistics
import sys
from dataclasses import dataclass
from typing import Optional

SOLVES_RE = re.compile(r"Policy solves (\d+) out of (\d+) problems")
WALL_TIME_RE = re.compile(r"Total wall time: ([\d.]+)s")


@dataclass(frozen=True)
class Job:
    domain: str
    domaindir: str
    seed: str
    config: str
    mem: str
    arm: str


@dataclass
class JobResult:
    solved: Optional[int] = None
    problems: Optional[int] = None
    wall_time: Optional[float] = None
    out_file: Optional[str] = None


def read_manifests(resdir: str) -> dict[str, list[Job]]:
    """arm -> ordered list of Jobs, one per manifest line (index = array task id - 1)."""
    jobs_by_arm: dict[str, list[Job]] = {}
    for path in sorted(glob.glob(os.path.join(resdir, "manifest-*.txt"))):
        arm = os.path.basename(path)[len("manifest-") : -len(".txt")]
        jobs = []
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                domain, domaindir, seed, config, mem, line_arm = line.split("|")
                jobs.append(Job(domain, domaindir, seed, config, mem, line_arm))
        jobs_by_arm[arm] = jobs
    return jobs_by_arm


def find_out_file(resdir: str, arm: str, task_id: int) -> Optional[str]:
    """The out file for array task `task_id` of `arm`, named "<TAG>-<arm>-<A>_<a>.out" by
    ablation.sh's `-o $RESDIR/out/%x-%A_%a.out" (%x = "<TAG>-<arm>", %A = job id, %a = task id).
    TAG is not otherwise known here, so match by suffix and the arm-delimiting hyphen instead."""
    pattern = re.compile(r"^.+-" + re.escape(arm) + r"-\d+_(\d+)\.out$")
    out_dir = os.path.join(resdir, "out")
    if not os.path.isdir(out_dir):
        return None
    for name in os.listdir(out_dir):
        m = pattern.match(name)
        if m and int(m.group(1)) == task_id:
            return os.path.join(out_dir, name)
    return None


def parse_out_file(path: str) -> JobResult:
    result = JobResult(out_file=path)
    text = open(path, errors="replace").read()
    m = SOLVES_RE.findall(text)
    if m:
        solved, problems = m[-1]
        result.solved = int(solved)
        result.problems = int(problems)
    w = WALL_TIME_RE.findall(text)
    if w:
        result.wall_time = float(w[-1])
    return result


def collect(resdir: str) -> tuple[dict[tuple[str, str, str], JobResult], list[tuple[str, str, str]]]:
    """Returns (results keyed by (domain, arm, seed), list of missing (domain, arm, seed))."""
    results: dict[tuple[str, str, str], JobResult] = {}
    missing: list[tuple[str, str, str]] = []
    jobs_by_arm = read_manifests(resdir)
    for arm, jobs in jobs_by_arm.items():
        for i, job in enumerate(jobs, start=1):
            key = (job.domain, arm, job.seed)
            out_file = find_out_file(resdir, arm, i)
            if out_file is None:
                missing.append(key)
                continue
            r = parse_out_file(out_file)
            if r.solved is None:
                missing.append(key)
            else:
                results[key] = r
    return results, missing


def domains_and_arms(results: dict[tuple[str, str, str], JobResult]) -> tuple[list[str], list[str]]:
    domains = sorted({d for d, _, _ in results})
    arms = sorted({a for _, a, _ in results})
    return domains, arms


def format_solved_cell(cells: list[JobResult]) -> str:
    if not cells:
        return "-"
    parts = [f"{c.solved}/{c.problems}" for c in cells]
    means = [c.solved for c in cells if c.solved is not None]
    mean_str = f"{statistics.mean(means):.1f}" if means else "-"
    return ", ".join(parts) + f" (mean {mean_str})"


def format_wall_cell(cells: list[JobResult]) -> str:
    times = [c.wall_time for c in cells if c.wall_time is not None]
    if not times:
        return "-"
    return f"{statistics.median(times):.0f}s"


def print_table(
    results: dict[tuple[str, str, str], JobResult],
    domains: list[str],
    arms: list[str],
    cell_fmt,
    title: str,
) -> None:
    print(f"\n## {title}\n")
    header = "| domain | " + " | ".join(arms) + " |"
    sep = "|---|" + "|".join(["---"] * len(arms)) + "|"
    print(header)
    print(sep)
    for domain in domains:
        row = [domain]
        for arm in arms:
            cells = [r for (d, a, s), r in sorted(results.items()) if d == domain and a == arm]
            row.append(cell_fmt(cells))
        print("| " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("resdir", help="results-* directory produced by scripts/ablation.sh")
    args = parser.parse_args()

    results, missing = collect(args.resdir)
    if not results and not missing:
        print(f"No manifest-*.txt files found in {args.resdir}", file=sys.stderr)
        sys.exit(1)
    domains, arms = domains_and_arms(results)
    # A domain/arm that is only ever missing still belongs in the header.
    for d, a, _ in missing:
        if d not in domains:
            domains.append(d)
        if a not in arms:
            arms.append(a)
    domains.sort()
    arms.sort()

    print(f"# {args.resdir}")
    print_table(results, domains, arms, format_solved_cell, "Solved / problems per seed (mean)")
    print_table(results, domains, arms, format_wall_cell, "Median wall time")

    if missing:
        print(f"\n## Missing ({len(missing)} of {len(results) + len(missing)})\n")
        print("No result (job still running, killed before logging, or not submitted):")
        for domain, arm, seed in sorted(missing):
            print(f"  - {domain} / {arm} / seed {seed}")
    else:
        print("\nNo gaps: every manifest line has a result.")


if __name__ == "__main__":
    main()
