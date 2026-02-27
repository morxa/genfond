import os

from genfond.topk_planner import compute_plans


def test_topk_planner():
    domain_dir = os.path.join(os.path.dirname(__file__), "fixtures", "pddl_files", "blocks3ops-det")
    domain_file = os.path.join(domain_dir, "domain.pddl")
    problem_file = os.path.join(domain_dir, "p004-2.pddl")
    plans = compute_plans(domain_file, problem_file, number_of_plans=3)
    assert len(plans) == 3
    assert plans[0] == [
        "(newtower b2 b3)",
        "(newtower b3 b0)",
        "(newtower b0 b1)",
        "(stack b1 b0)",
        "(stack b2 b1)",
        "(stack b3 b2)",
    ]
    assert plans[1] == [
        "(newtower b2 b3)",
        "(newtower b3 b0)",
        "(move b0 b1 b2)",
        "(newtower b0 b2)",
        "(stack b1 b0)",
        "(stack b2 b1)",
        "(stack b3 b2)",
    ]
    assert plans[2] == [
        "(newtower b2 b3)",
        "(move b3 b0 b2)",
        "(newtower b0 b1)",
        "(newtower b3 b2)",
        "(stack b1 b0)",
        "(stack b2 b1)",
        "(stack b3 b2)",
    ]
