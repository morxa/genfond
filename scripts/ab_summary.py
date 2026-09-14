"""Summarise one or more results-* directories produced by scripts/ab_bench.sh.

    python scripts/ab_summary.py results-...-baseline [results-...-hypothesis]

Prints one row per (suite, type): solved/problems, wall time, policy cost, peak memory, and how
often the run hit clingo's ground-fact id ceiling ("Id out of range") or the memory cap.
"""

import csv
import glob
import os
import re
import sys


def load(resdir):
    rows = {}
    stats = os.path.join(resdir, "stats.csv")
    if os.path.exists(stats):
        with open(stats) as f:
            for r in csv.DictReader(f):
                rows[(r["domain"], r["constraintType"])] = r
    for log in glob.glob(os.path.join(resdir, "out", "*.log")):
        name = os.path.basename(log)[: -len(".log")]
        m = re.match(r"(.*)-(datalog(?:-[a-z-]+)?|state|trans|d2l|exact)$", name)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        text = open(log, errors="replace").read()
        r = rows.setdefault(key, {"domain": key[0], "constraintType": key[1]})
        r["idOutOfRange"] = len(re.findall(r"[Ii]d out of range", text))
        r["memoryErrors"] = len(re.findall(r"MemoryError|OUT_OF_RESOURCES|std::bad_alloc", text))
        m2 = re.findall(r"Policy solves (\d+) out of (\d+)", text)
        if m2 and "solved" not in r:
            r["solved"], r["problems"] = m2[-1]
            r["note"] = "from log"
    exits = os.path.join(resdir, "exit-codes.txt")
    if os.path.exists(exits):
        for line in open(exits):
            m = re.match(r"(.*)-(datalog(?:-[a-z-]+)?|state|trans|d2l|exact) rc=(\d+) wall=(\d+)s", line)
            if m:
                r = rows.setdefault((m.group(1), m.group(2)), {"domain": m.group(1), "constraintType": m.group(2)})
                r["rc"] = m.group(3)
                r.setdefault("totalWallTime", m.group(4))
    return rows


def fmt(r, k, f="{}"):
    v = r.get(k)
    if v in (None, ""):
        return "-"
    try:
        return f.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def main():
    for resdir in sys.argv[1:]:
        rows = load(resdir)
        print(f"\n{resdir}")
        header = f"{'suite':28} {'type':12} {'solved':>8} {'wall':>8} {'cost':>5} {'mem MB':>8} {'idOOR':>5} {'memErr':>6} rc"
        print(header)
        print("-" * len(header))
        for (suite, ptype), r in sorted(rows.items()):
            solved = f"{fmt(r, 'solved', '{:.0f}')}/{fmt(r, 'problems', '{:.0f}')}"
            print(
                f"{suite:28} {ptype:12} {solved:>8} {fmt(r, 'totalWallTime', '{:.0f}s'):>8} "
                f"{fmt(r, 'cost', '{:.0f}'):>5} {fmt(r, 'memUsage', '{:.0f}'):>8} "
                f"{fmt(r, 'idOutOfRange'):>5} {fmt(r, 'memoryErrors'):>6} {r.get('rc', '-')}"
            )


if __name__ == "__main__":
    main()
