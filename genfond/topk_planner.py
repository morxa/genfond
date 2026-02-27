from pddl.parser.plan import PlanParser
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import AnytimePlanner


def action_to_lisp_str(action):
    return f"({action.action.name} {' '.join([str(p) for p in action.actual_parameters])})"


def compute_plans(domain_file, problem_file, number_of_plans=3):
    reader = PDDLReader()
    plan_parser = PlanParser()
    problem = reader.parse_problem(domain_file, problem_file)
    plans = []
    with AnytimePlanner(name="symk", params={"number_of_plans": number_of_plans}) as planner:
        for i, result in enumerate(planner.get_solutions(problem)):
            if result.status == PlanGenerationResultStatus.INTERMEDIATE:
                plan_str = " ".join([action_to_lisp_str(action) for action in result.plan.actions])
                plans.append(plan_parser(plan_str))
    return plans
