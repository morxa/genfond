"""H26: checkpoint the best-coverage policy and a provisional stats row immediately, so a hard
kill (e.g. SLURM's SIGKILL following the graceful SIGTERM warning) cannot lose them.

These mock out `solve_step`/`execute_policy` exactly like `tests/test_final_cost_minimization.py`
and `tests/test_optimal_model_enumeration.py`, so the round loop's own control flow is exercised
without any of the feature-generation/clingo machinery underneath -- see AGENTS.md's Tests
section on preferring the fast, raw fixtures.
"""

import csv
import os
import pickle

import genfond.iterative_solver as isolver
from genfond.checkpoint import append_stats_row, atomic_pickle_dump, checkpoint_best_policy, remove_provisional_row
from genfond.problem_iterator import Result
from genfond.shutdown import request_stop, reset_stop


class DummyProblem:
    def __init__(self, name, objects=()):
        self.name = name
        self.objects = objects


class FakePolicy:
    def __init__(self, name, cost=(1,)):
        self.name = name
        self.cost = cost

    def __repr__(self):
        return f"FakePolicy({self.name})"


def _config(**overrides):
    config = {
        "min_complexity": 2,
        "max_complexity": 6,
        "use_unrestricted_features": False,
        "reset_complexity_on_state_space_change": False,
        "add_problem_after_success": True,
        "unselect_problems": False,
        "min_number_of_plans": 5,
        "max_frontier_expansions": 20,
        "use_example_plans": False,
        "frontier_expansion": False,
        "policy_iterations": 1,
        "validation_iterations": 1,
        "stop_after_first_solution": True,
        "final_cost_minimization": False,
        "minimize_good_signatures": "none",
        "minimize_selected_count": "none",
        "checkpoint_best_policy": True,
    }
    config.update(overrides)
    return config


# --- checkpoint_best_policy / atomic_pickle_dump -----------------------------------------------


def test_atomic_pickle_dump_leaves_no_tmp_file_behind(tmp_path):
    path = str(tmp_path / "out.policy")
    atomic_pickle_dump(FakePolicy("p"), path)

    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
    with open(path, "rb") as f:
        assert pickle.load(f).name == "p"


def test_checkpoint_best_policy_writes_when_enabled_and_output_configured(tmp_path, caplog):
    path = str(tmp_path / "out.policy")
    policy = FakePolicy("best")

    with caplog.at_level("INFO", logger="genfond.checkpoint"):
        checkpoint_best_policy(policy, 3, 5, {"checkpoint_best_policy": True, "output": path})

    with open(path, "rb") as f:
        assert pickle.load(f).name == "best"
    assert any("Checkpointed best policy (3/5)" in message for message in caplog.messages)


def test_checkpoint_best_policy_noop_without_output_path(tmp_path):
    checkpoint_best_policy(FakePolicy("best"), 1, 1, {"checkpoint_best_policy": True, "output": None})

    assert list(tmp_path.iterdir()) == []


def test_checkpoint_best_policy_noop_when_disabled(tmp_path):
    path = str(tmp_path / "out.policy")

    checkpoint_best_policy(FakePolicy("best"), 1, 1, {"checkpoint_best_policy": False, "output": path})

    assert not os.path.exists(path)


def test_checkpoint_best_policy_default_is_enabled(tmp_path):
    """`checkpoint_best_policy` defaults to on when the key is absent -- config.get(..., True) is
    the enforced default, matching genfond/config/default.yaml."""
    path = str(tmp_path / "out.policy")

    checkpoint_best_policy(FakePolicy("best"), 1, 1, {"output": path})

    assert os.path.exists(path)


# --- solve_iteratively: checkpoint appears immediately after each coverage improvement ---------


def test_checkpoint_file_tracks_best_coverage_through_the_round_loop(monkeypatch, tmp_path):
    """Round 1 only solves p1; round 2 (p2 added by add_problem_after_success) solves both.
    The checkpoint on disk must already hold the round-1 policy by the time round 2 starts (not
    only once the whole run finishes), and end up equal to the policy solve_iteratively returns.
    """
    p1, p2 = DummyProblem("p1", objects=[1]), DummyProblem("p2", objects=[1, 2])
    problems = [p1, p2]
    policy1 = FakePolicy("first", cost=(3,))
    policy2 = FakePolicy("second", cost=(2,))
    output_path = tmp_path / "out.policy"

    calls = {"n": 0}

    def fake_solve_step(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return Result.SUCCESS, policy1, []
        # Round 2 must see round 1's improvement already checkpointed -- immediately, not only
        # after solve_iteratively itself returns.
        with open(output_path, "rb") as f:
            assert pickle.load(f).name == policy1.name
        return Result.SUCCESS, policy2, []

    def fake_execute_policy(domain, problem, policy, config, **kwargs):
        if policy is policy1 and problem.name == "p2":
            raise RuntimeError("no action found")
        return []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", fake_execute_policy)

    config = _config(output=str(output_path))
    policy, solved, stats = isolver.solve_iteratively(None, problems, config)

    assert calls["n"] == 2
    assert policy is policy2
    assert {p.name for p in solved} == {"p1", "p2"}
    with open(output_path, "rb") as f:
        final_checkpoint = pickle.load(f)
    assert final_checkpoint.name == policy.name
    assert not os.path.exists(str(output_path) + ".tmp")


def test_checkpoint_not_written_when_disabled(monkeypatch, tmp_path):
    p1 = DummyProblem("p1", objects=[1])
    output_path = tmp_path / "out.policy"

    def fake_solve_step(**kwargs):
        return Result.SUCCESS, FakePolicy("only"), []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda *a, **k: [])

    config = _config(output=str(output_path), checkpoint_best_policy=False)
    isolver.solve_iteratively(None, [p1], config)

    assert not output_path.exists()


def test_checkpoint_not_written_without_output_configured(monkeypatch, tmp_path):
    p1 = DummyProblem("p1", objects=[1])

    def fake_solve_step(**kwargs):
        return Result.SUCCESS, FakePolicy("only"), []

    monkeypatch.setattr(isolver, "solve_step", fake_solve_step)
    monkeypatch.setattr(isolver, "execute_policy", lambda *a, **k: [])

    config = _config(output=None)
    isolver.solve_iteratively(None, [p1], config)

    assert list(tmp_path.iterdir()) == []


# --- provisional stats row: append_stats_row / remove_provisional_row --------------------------


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_append_stats_row_creates_the_file_with_a_header(tmp_path):
    path = str(tmp_path / "stats.csv")
    append_stats_row(path, {"domain": "d", "solved": 3})

    rows = _read_rows(path)
    assert len(rows) == 1
    assert rows[0] == {"domain": "d", "solved": "3"}


def test_append_stats_row_skips_a_missing_file_when_create_if_missing_is_false(tmp_path):
    """The provisional write's guard: a brand new stats file must not get its header
    established from a provisional row's much smaller key set, since that would then truncate
    every later (including this run's own final) row for the rest of the file's life."""
    path = str(tmp_path / "stats.csv")

    append_stats_row(path, {"domain": "d", "provisional": 1}, create_if_missing=False)

    assert not os.path.exists(path)


def test_append_stats_row_follows_an_existing_header_and_drops_unknown_keys(tmp_path, caplog):
    path = str(tmp_path / "stats.csv")
    append_stats_row(path, {"domain": "d", "solved": 1})

    with caplog.at_level("WARNING"):
        append_stats_row(path, {"domain": "d", "solved": 2, "newColumn": "x"})

    rows = _read_rows(path)
    assert len(rows) == 2
    assert "newColumn" not in rows[1]
    assert any("newColumn" in message for message in caplog.messages)


def test_remove_provisional_row_deletes_only_the_matching_run(tmp_path):
    path = str(tmp_path / "stats.csv")
    append_stats_row(path, {"runId": "run-a", "provisional": 1, "solved": 1})
    append_stats_row(path, {"runId": "run-b", "provisional": 1, "solved": 2})

    remove_provisional_row(path, "run-a")

    rows = _read_rows(path)
    assert [row["runId"] for row in rows] == ["run-b"]


def test_remove_provisional_row_noop_when_nothing_matches(tmp_path):
    path = str(tmp_path / "stats.csv")
    append_stats_row(path, {"runId": "run-a", "provisional": 1})

    remove_provisional_row(path, "no-such-run")

    rows = _read_rows(path)
    assert len(rows) == 1


def test_remove_provisional_row_noop_on_a_missing_file(tmp_path):
    # Must not raise even though the file was never created.
    remove_provisional_row(str(tmp_path / "does-not-exist.csv"), "run-a")


def test_provisional_row_then_final_row_leaves_exactly_one_row(tmp_path):
    """The sequence __main__.py runs at exit: delete this run's provisional row (if any), then
    append the real final row -- a run that finishes normally must never leave a duplicate."""
    path = str(tmp_path / "stats.csv")
    provisional = {"runId": "run-a", "provisional": 1, "domain": "d", "stoppedBy": "signal"}
    append_stats_row(path, provisional, create_if_missing=False)
    assert not os.path.exists(path)  # the guard above: no file existed yet, so nothing was written

    # Now simulate the file already existing (e.g. from an earlier run) so the provisional write
    # actually lands, then run the finish sequence for "run-a".
    append_stats_row(path, {"runId": "run-0", "provisional": 0, "domain": "d", "stoppedBy": ""})
    append_stats_row(path, provisional, create_if_missing=False)
    assert len(_read_rows(path)) == 2

    final = {"runId": "run-a", "provisional": 0, "domain": "d", "stoppedBy": ""}
    remove_provisional_row(path, "run-a")
    append_stats_row(path, final)

    rows = _read_rows(path)
    assert [row["runId"] for row in rows] == ["run-0", "run-a"]
    assert rows[1]["provisional"] == "0"


# --- solve_iteratively: a graceful stop writes a provisional row immediately -------------------


def test_stop_requested_writes_a_provisional_stats_row(monkeypatch, tmp_path):
    p1 = DummyProblem("p1", objects=[1])
    stats_path = tmp_path / "stats.csv"
    # The provisional writer skips a brand-new file (see the guard test above), so seed one
    # first, as a real shared benchmark stats file normally already exists (AGENTS.md's A/B
    # benchmark protocol).
    append_stats_row(str(stats_path), {"domain": "seed", "runId": "seed", "provisional": 0, "stoppedBy": ""})

    calls: list = []
    monkeypatch.setattr(isolver, "solve_step", lambda **kwargs: calls.append(kwargs) or (Result.NO_SOLUTION, None, []))

    config = _config(stats=str(stats_path))
    try:
        request_stop()
        policy, solved, stats = isolver.solve_iteratively(None, [p1], config)
    finally:
        reset_stop()

    assert calls == []
    assert stats["stoppedBy"] == "signal"
    rows = _read_rows(str(stats_path))
    provisional_rows = [row for row in rows if row["runId"] == stats["runId"]]
    assert len(provisional_rows) == 1
    assert provisional_rows[0]["provisional"] == "1"
    assert provisional_rows[0]["stoppedBy"] == "signal"


def test_stop_requested_does_not_write_a_provisional_row_when_disabled(monkeypatch, tmp_path):
    p1 = DummyProblem("p1", objects=[1])
    stats_path = tmp_path / "stats.csv"
    append_stats_row(str(stats_path), {"domain": "seed", "runId": "seed", "provisional": 0, "stoppedBy": ""})

    monkeypatch.setattr(isolver, "solve_step", lambda **kwargs: (Result.NO_SOLUTION, None, []))

    config = _config(stats=str(stats_path), checkpoint_best_policy=False)
    try:
        request_stop()
        policy, solved, stats = isolver.solve_iteratively(None, [p1], config)
    finally:
        reset_stop()

    rows = _read_rows(str(stats_path))
    assert len(rows) == 1  # only the seed row


def test_stop_requested_does_nothing_without_a_stats_path(monkeypatch, tmp_path):
    p1 = DummyProblem("p1", objects=[1])
    monkeypatch.setattr(isolver, "solve_step", lambda **kwargs: (Result.NO_SOLUTION, None, []))

    config = _config(stats=None)
    try:
        request_stop()
        isolver.solve_iteratively(None, [p1], config)
    finally:
        reset_stop()

    assert list(tmp_path.iterdir()) == []
