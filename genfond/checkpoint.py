"""Checkpointing the best-coverage policy and a provisional stats row (H26).

`iterative_solver.solve_iteratively` only returns its result -- the best policy `keep_best_policy`
tracked and the stats dict -- at the very end of a run, and `__main__.py` pickles the policy and
appends the stats row only after that. A graceful stop (SIGTERM -> genfond.shutdown.request_stop)
is polled between rounds and inside a single clingo solve (see `Solver.solve`), but *grounding*
(`Solver.__init__`'s `self.control.ground(parts)`) is one blocking call with no such poll point.
When grounding alone outlasts the SLURM time limit's reserve, the hard `SIGKILL` that follows the
warning `SIGTERM` arrives before `solve_iteratively` ever returns, and both the best policy and
the stats row are lost even though earlier rounds already made progress.

This module gives the run two independent, immediate write-backs so that progress survives a
kill at any point, gated by the single `checkpoint_best_policy` config flag (default on):

* `checkpoint_best_policy` -- pickled atomically (write `<path>.tmp`, then `os.replace`) to the
  configured `output` path every time the best-coverage policy improves, i.e. from inside the
  round loop itself, long before the run would otherwise finish. A reader (e.g.
  `scripts/eval_policy.py`) always sees either the previous complete pickle or the new one, never
  a half-written file.
* `append_stats_row` / `remove_provisional_row` -- the same CSV writer `__main__.py` uses for the
  final row, factored out so a *provisional* row (tagged with a per-run `runId`) can be written
  the moment a graceful stop is noticed inside the round loop, and then deleted (by `runId`, under
  the same `FileLock`) right before the final row for that run is appended -- so a run that
  actually finishes never leaves a duplicate behind, and a run that is killed before finishing
  leaves exactly the provisional row.

Neither mechanism can do anything about a kill that lands *during* the blocking `ground()` call
itself -- Python cannot run a signal handler while control is inside that call -- so this only
closes the gap between successive rounds (and the final verification phase in `__main__`), not a
single grounding call that itself outlasts the reserve. What it does provide: the *previous*
round's best policy and a provisional stats row are already on disk before that risky call even
starts.
"""

import csv
import logging
import os
import pickle
from typing import Any, Mapping, Optional

from filelock import FileLock

log = logging.getLogger("genfond.checkpoint")

# Column used to identify a run's own row across the provisional write and the final write, so
# the latter can delete the former instead of leaving a duplicate. Also usable, in combination
# with "name" and "startTime", as the simpler fallback the task description allows -- but a
# dedicated random id sidesteps any timestamp-collision/precision concerns, so it is used as the
# sole matching key here.
RUN_ID_COLUMN = "runId"
PROVISIONAL_COLUMN = "provisional"


def atomic_pickle_dump(obj: Any, path: str) -> None:
    """Pickle `obj` to `path` without ever leaving a partially-written file there.

    Writes to `<path>.tmp` first and `os.replace`s it into place -- `os.replace` is atomic on the
    same filesystem, so a reader of `path` (e.g. `scripts/eval_policy.py`, or a later checkpoint
    in this same run) always sees either the previous complete file or the new one.
    """
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def checkpoint_best_policy(policy: Any, num_solved: int, total: int, config: Mapping) -> None:
    """Pickle `policy` to `config["output"]` immediately, if checkpointing is enabled.

    Called every time the best-coverage policy tracked by `keep_best_policy` improves. A no-op
    when `checkpoint_best_policy` is off (config gate) or no `output` path is configured -- there
    is nothing to checkpoint to.
    """
    if not config.get("checkpoint_best_policy", True):
        return
    output_path = config.get("output")
    if not output_path:
        return
    atomic_pickle_dump(policy, output_path)
    log.info(f"Checkpointed best policy ({num_solved}/{total}) to {output_path}")


def _read_header(path: str) -> Optional[list[str]]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        try:
            return next(csv.reader(f))
        except StopIteration:
            return None


def append_stats_row(path: str, stats: Mapping[str, Any], create_if_missing: bool = True) -> None:
    """Append one row to the stats CSV at `path`, locked with `<path>.lock`.

    Follows the existing file's header (dropping -- with a warning -- any of `stats`'s keys that
    are not in it, exactly as `__main__.py`'s final write always has, since the key set differs
    between runs: e.g. `failureReason` only exists on failure). When the file does not exist yet:
    `create_if_missing` (the final write's case) creates it with a header taken from `stats`
    itself; the provisional write passes `create_if_missing=False` instead, since a header
    established from a provisional row's much smaller key set (it is written mid-run, before
    totalWallTime/cost/memUsage/... are known) would then truncate every later row -- including
    this run's own final one -- for the rest of the file's life. Skipping the provisional write
    on a brand new file only loses the checkpoint for the very first run ever appended to a given
    stats file, which is the uncommon case in practice (see AGENTS.md's benchmark protocol: a
    stats file is normally shared across many runs).
    """
    lock = FileLock(path + ".lock")
    with lock:
        file_exists = os.path.isfile(path)
        if not file_exists and not create_if_missing:
            log.debug("Not writing a provisional stats row: %s does not exist yet", path)
            return
        fieldnames = _read_header(path) if file_exists else list(stats.keys())
        assert fieldnames is not None
        dropped = set(stats) - set(fieldnames)
        if dropped:
            log.warning(f"Stats keys not in the header of {path}, dropped: {sorted(dropped)}")
        with open(path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)


def remove_provisional_row(path: str, run_id: str) -> None:
    """Delete the provisional row tagged with `run_id` (if any) from the stats CSV at `path`.

    Called right before the final row for the same run is appended (see `append_stats_row`), so
    that a run which finishes normally never leaves the provisional row behind as a duplicate.
    A no-op if the file does not exist, has no `RUN_ID_COLUMN` in its header (e.g. it predates
    this feature), or has no row tagged with `run_id`.
    """
    if not os.path.isfile(path):
        return
    lock = FileLock(path + ".lock")
    with lock:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if fieldnames is None or RUN_ID_COLUMN not in fieldnames:
                return
            rows = [row for row in reader if row.get(RUN_ID_COLUMN) != run_id]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
