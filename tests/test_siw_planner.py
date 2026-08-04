import os

import pddl

from genfond.ground import action_string
from genfond.siw_planner import compute_plans

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "pddl_files")

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

# (g) is only reachable via a state satisfying both (p) and (q), which no width-1 search
# reaches: restore-pq is the only action adding both, and neither of its effect atoms is
# novel on its own by the time it becomes applicable.
WIDTH_TWO_DOMAIN_TEXT = """(define (domain needs-width-two)
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

WIDTH_TWO_PROBLEM_TEXT = """(define (problem needs-width-two-instance)
  (:domain needs-width-two)
  (:init)
  (:goal (and (g)))
)
"""


def siw_config(**overrides):
    return {"algorithm": "siw", "max_width": 2, "prune": True} | overrides


def read_fixture(*parts):
    with open(os.path.join(FIXTURE_DIR, *parts)) as fixture_file:
        return fixture_file.read()


def plan_action_strings(plan, domain):
    return [action_string(action) for action in plan.instantiate(domain)]


def test_compute_plans_uses_topk_plan_syntax(tmp_path):
    domain_path = tmp_path / "domain.pddl"
    domain_path.write_text(DOMAIN_TEXT)
    domain = pddl.parse_domain(str(domain_path))

    plans = list(compute_plans(DOMAIN_TEXT, PROBLEM_TEXT, siw_config()))

    assert plans
    assert plan_action_strings(plans[0], domain) == ["finish()"]


def test_compute_plans_yields_a_single_plan():
    plans = list(compute_plans(WIDTH_TWO_DOMAIN_TEXT, WIDTH_TWO_PROBLEM_TEXT, siw_config()))

    assert len(plans) == 1


def test_compute_plans_respects_configured_width_limit(tmp_path):
    domain_path = tmp_path / "needs-width-two-domain.pddl"
    domain_path.write_text(WIDTH_TWO_DOMAIN_TEXT)
    domain = pddl.parse_domain(str(domain_path))

    assert list(compute_plans(WIDTH_TWO_DOMAIN_TEXT, WIDTH_TWO_PROBLEM_TEXT, siw_config(max_width=1))) == []

    plans = list(compute_plans(WIDTH_TWO_DOMAIN_TEXT, WIDTH_TWO_PROBLEM_TEXT, siw_config(max_width=2)))
    assert plan_action_strings(plans[0], domain) == ["set-p()", "set-q()", "restore-pq()", "finish()"]


def test_compute_plans_supports_plain_iterated_width(tmp_path):
    domain_path = tmp_path / "needs-width-two-domain.pddl"
    domain_path.write_text(WIDTH_TWO_DOMAIN_TEXT)
    domain = pddl.parse_domain(str(domain_path))

    plans = list(compute_plans(WIDTH_TWO_DOMAIN_TEXT, WIDTH_TWO_PROBLEM_TEXT, siw_config(algorithm="iw")))

    assert plan_action_strings(plans[0], domain) == ["set-p()", "set-q()", "restore-pq()", "finish()"]


def test_compute_plans_returns_nothing_for_unsupported_pddl():
    """SIW is a classical planner; a FOND domain must degrade quietly, not raise."""
    domain_text = read_fixture("blocksworld-fond", "domain.pddl")
    problem_text = read_fixture("blocksworld-fond", "p01.pddl")

    assert list(compute_plans(domain_text, problem_text, siw_config())) == []


def test_compute_plans_returns_nothing_for_unsolvable_problem():
    # handempty is absent from (:init), so no action is ever applicable.
    domain_text = read_fixture("typed-blocks", "domain.pddl")
    problem_text = read_fixture("typed-blocks", "p01.pddl")

    assert list(compute_plans(domain_text, problem_text, siw_config())) == []


def test_compute_plans_returns_nothing_when_goal_already_holds():
    problem_text = """(define (problem reach-goal-instance)
  (:domain reach-goal)
  (:init (done))
  (:goal (and (done)))
)
"""

    assert list(compute_plans(DOMAIN_TEXT, problem_text, siw_config())) == []
