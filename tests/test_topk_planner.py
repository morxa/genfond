import os

import pddl

from genfond.ground import action_string
from genfond.topk_planner import compute_plans

from .helpers import get_action


def test_topk_planner():
    domain_dir = os.path.join(os.path.dirname(__file__), "fixtures", "pddl_files", "blocks3ops-det")
    domain_file = os.path.join(domain_dir, "domain.pddl")
    problem_file = os.path.join(domain_dir, "p004-2.pddl")
    domain = pddl.parse_domain(domain_file)
    with open(domain_file, "r") as f:
        domain_str = f.read()
    with open(problem_file, "r") as f:
        problem_str = f.read()
    plans = compute_plans(domain_str, problem_str, number_of_plans=3)
    assert len(plans) == 3
    str_plans = [[action_string(a) for a in plan.instantiate(domain)] for plan in plans]
    assert [
        "newtower(b2,b3)",
        "newtower(b3,b0)",
        "newtower(b0,b1)",
        "stack(b1,b0)",
        "stack(b2,b1)",
        "stack(b3,b2)",
    ] in str_plans
    assert [
        "newtower(b2,b3)",
        "newtower(b3,b0)",
        "move(b0,b1,b2)",
        "newtower(b0,b2)",
        "stack(b1,b0)",
        "stack(b2,b1)",
        "stack(b3,b2)",
    ] in str_plans
    assert [
        "newtower(b2,b3)",
        "move(b3,b0,b2)",
        "newtower(b0,b1)",
        "newtower(b3,b2)",
        "stack(b1,b0)",
        "stack(b2,b1)",
        "stack(b3,b2)",
    ] in str_plans
