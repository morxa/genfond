"""Process-wide cooperative shutdown flag.

`request_stop()` is called from the signal handler installed in `genfond.__main__` for both
SIGINT and SIGTERM (SLURM sends SIGTERM ahead of the hard kill when a job's time limit is
reached; see `genfond.bash`'s `--signal=B:TERM@600`). `stop_requested()` is polled from two
places that otherwise have no way to hear about an external signal in a timely fashion:

* `iterative_solver.solve_iteratively`'s round loop, so it stops starting new rounds instead of
  running the escalation ladder to completion.
* `Solver.solve`'s async wait loop, so a clingo solve already in flight gets cancelled instead
  of running to completion -- a signal delivered while a blocking C call holds control would
  otherwise only be seen by Python once that call returns on its own.

A single process-wide flag is enough: once shutdown is requested there is nothing to resume, and
`genfond.__main__` is the only place that starts a run.
"""

import threading

_stop_event = threading.Event()


def request_stop() -> None:
    _stop_event.set()


def stop_requested() -> bool:
    return _stop_event.is_set()


def reset_stop() -> None:
    """Test helper: clear the flag so one test's signal doesn't leak into the next."""
    _stop_event.clear()
