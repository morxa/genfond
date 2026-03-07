from typing import Iterator

import unified_planning
from pddl.parser.plan import Plan, PlanParser
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import AnytimePlanner

unified_planning.shortcuts.get_environment().credits_stream = None


def action_to_lisp_str(action):
    return f"({action.action.name} {' '.join([str(p) for p in action.actual_parameters])})"


def compute_plans(domain_str: str, problem_str: str, number_of_plans: int = 3) -> Iterator[Plan]:
    reader = PDDLReader()
    plan_parser = PlanParser()
    problem = reader.parse_problem_string(domain_str, problem_str)
    with AnytimePlanner(name="symk", params={"number_of_plans": number_of_plans}) as planner:
        for result in planner.get_solutions(problem):
            if result.status == PlanGenerationResultStatus.INTERMEDIATE:
                plan_str = " ".join([action_to_lisp_str(action) for action in result.plan.actions])
                yield plan_parser(plan_str)
