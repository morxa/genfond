import pddl
import pytest

from genfond.ground import action_string
from genfond.iw import (
    IWPlanner,
    compute_plans,
    load_problem,
    set_compute_plans_max_width,
    solve_iw,
)

DOMAIN_TEXT = """(define (domain reach-goal)
  (:requirements :strips)
  (:predicates (done))
  (:action finish
    :parameters ()
    :precondition (and)
    :effect (and (done))
  )
)
"""

PROBLEM_TEXT = """(define (problem reach-goal-instance)
  (:domain reach-goal)
  (:init)
  (:goal (and (done)))
)
"""


def write_task_files(tmp_path):
    domain_path = tmp_path / "domain.pddl"
    problem_path = tmp_path / "problem.pddl"
    domain_path.write_text(DOMAIN_TEXT)
    problem_path.write_text(PROBLEM_TEXT)
    return domain_path, problem_path


def test_load_problem_parses_pddl_files(tmp_path):
    domain_path, problem_path = write_task_files(tmp_path)

    parsed_problem = load_problem(domain_path, problem_path)

    assert parsed_problem.domain.get_name() == "reach-goal"
    assert parsed_problem.problem.get_name() == "reach-goal-instance"
    assert parsed_problem.initial_state.get_problem().get_index() == parsed_problem.problem.get_index()


def test_solve_iw_returns_single_action_plan(tmp_path):
    domain_path, problem_path = write_task_files(tmp_path)

    result = solve_iw(domain_path, problem_path, width=1)

    assert result.status == "solved"
    assert result.solution is not None
    assert len(result.solution) == 1
    assert result.solution[0].get_action().get_name() == "finish"
    assert result.solution_cost == 1.0


def test_iw_planner_rejects_non_positive_width():
    with pytest.raises(ValueError, match="width must be positive"):
        IWPlanner(0)


def test_compute_plans_uses_topk_plan_syntax(tmp_path):
    set_compute_plans_max_width(1)
    domain_path, _ = write_task_files(tmp_path)
    domain = pddl.parse_domain(str(domain_path))

    plans = list(compute_plans(DOMAIN_TEXT, PROBLEM_TEXT, number_of_plans=3))

    assert plans
    assert [action_string(action) for action in plans[0].instantiate(domain)] == ["finish()"]


def test_compute_plans_returns_multiple_solutions_for_same_width(tmp_path):
    set_compute_plans_max_width(1)
    domain_text = """(define (domain two-solutions)
  (:requirements :strips)
  (:predicates (g))
  (:action finish-a
    :parameters ()
    :precondition (and)
    :effect (and (g))
  )
  (:action finish-b
    :parameters ()
    :precondition (and)
    :effect (and (g))
  )
)
"""
    problem_text = """(define (problem two-solutions-instance)
  (:domain two-solutions)
  (:init)
  (:goal (and (g)))
)
"""
    domain_path = tmp_path / "two-solutions-domain.pddl"
    domain_path.write_text(domain_text)
    domain = pddl.parse_domain(str(domain_path))

    plans = list(compute_plans(domain_text, problem_text, number_of_plans=10))
    str_plans = sorted(" ".join(action_string(action) for action in plan.instantiate(domain)) for plan in plans)

    assert str_plans == ["finish-a()", "finish-b()"]


def test_compute_plans_respects_configured_width_limit():
    domain_text = """(define (domain needs-width-two)
  (:requirements :strips)
  (:predicates (p) (q) (g))
  (:action set-p
    :parameters ()
    :precondition (and)
    :effect (and (p))
  )
  (:action set-q
    :parameters ()
    :precondition (and (p))
    :effect (and (q) (not (p)))
  )
  (:action restore-pq
    :parameters ()
    :precondition (and (q))
    :effect (and (p) (q))
  )
  (:action finish
    :parameters ()
    :precondition (and (p) (q))
    :effect (and (g))
  )
)
"""
    problem_text = """(define (problem needs-width-two-instance)
  (:domain needs-width-two)
  (:init)
  (:goal (and (g)))
)
"""

    set_compute_plans_max_width(1)
    assert list(compute_plans(domain_text, problem_text, number_of_plans=3)) == []

    set_compute_plans_max_width(2)
    assert list(compute_plans(domain_text, problem_text, number_of_plans=3))

    set_compute_plans_max_width(1)
