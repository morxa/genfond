"""--max-wall-time / SIGTERM-SIGINT: solve_iteratively must stop starting new rounds once the
wall-clock budget is exhausted or a stop has been requested, without touching the (expensive)
solve machinery at all.

`solve_step` is monkeypatched out here rather than exercised for real: the gate this tests sits
at the very top of the round loop, before `solve_step` is ever called, so a fake that just
records its calls is enough to prove the loop breaks before starting a round -- and keeps these
tests fast, in line with this repo's preference for the raw / low-level fixtures over full
end-to-end solves (see AGENTS.md's Tests section).
"""

import genfond.iterative_solver as iterative_solver
from genfond.config_handler import ConfigHandler
from genfond.iterative_solver import solve_iteratively
from genfond.problem_iterator import Result
from genfond.shutdown import request_stop, reset_stop


def _no_op_solve_step(calls):
    def fake(**kwargs):
        calls.append(kwargs)
        return Result.NO_SOLUTION, None, []

    return fake


def test_an_already_exhausted_wall_budget_stops_before_the_first_round(monkeypatch, simple_blocks):
    domain, problem = simple_blocks
    calls: list = []
    monkeypatch.setattr(iterative_solver, "solve_step", _no_op_solve_step(calls))
    config = ConfigHandler(type="d2l")
    config["max_wall_time"] = 0.0
    config["wall_time_reserve"] = 0

    policy, succs, stats = solve_iteratively(domain, [problem], config)

    assert calls == []
    assert policy is None
    assert stats["stoppedBy"] == "wall_time"
    assert stats["wallBudgetUsed"] >= 0.0
    # __main__ reads stats["totalSolveCpuTime"] unconditionally (see iterative_solver.py); it is
    # normally seeded by solve_step's first round, which never runs on this path.
    assert stats["totalSolveCpuTime"] == 0


def test_a_generous_wall_budget_does_not_stop_the_first_round(monkeypatch, simple_blocks):
    domain, problem = simple_blocks
    calls: list = []
    monkeypatch.setattr(iterative_solver, "solve_step", _no_op_solve_step(calls))
    config = ConfigHandler(type="d2l")
    config["max_wall_time"] = 3600
    config["wall_time_reserve"] = 0

    solve_iteratively(domain, [problem], config)

    assert len(calls) >= 1
    assert calls[0]["wall_deadline"] is not None


def test_max_wall_time_unset_never_touches_the_wall_deadline(monkeypatch, simple_blocks):
    """Default behaviour (max_wall_time: null) must be byte-identical: wall_deadline is None."""
    domain, problem = simple_blocks
    calls: list = []
    monkeypatch.setattr(iterative_solver, "solve_step", _no_op_solve_step(calls))
    config = ConfigHandler(type="d2l")
    assert config["max_wall_time"] is None

    _, _, stats = solve_iteratively(domain, [problem], config)

    assert len(calls) >= 1
    assert all(call["wall_deadline"] is None for call in calls)
    assert "stoppedBy" not in stats
    assert "wallBudgetUsed" not in stats


def test_a_pending_stop_request_stops_before_the_first_round(monkeypatch, simple_blocks):
    domain, problem = simple_blocks
    calls: list = []
    monkeypatch.setattr(iterative_solver, "solve_step", _no_op_solve_step(calls))
    config = ConfigHandler(type="d2l")
    try:
        request_stop()
        policy, succs, stats = solve_iteratively(domain, [problem], config)
        assert calls == []
        assert policy is None
        assert stats["stoppedBy"] == "signal"
        assert stats["totalSolveCpuTime"] == 0
    finally:
        reset_stop()
