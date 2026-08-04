import logging
from typing import Any, Iterator, Sequence

import siw
from pddl.core import Plan
from pddl.logic.terms import Constant
from pddl.parser.domain import DomainParser
from pddl.parser.problem import ProblemParser

log = logging.getLogger("genfond.siw_planner")


def to_pddl_plan(plan: Sequence[siw.GroundAction]) -> Plan:
    """Convert a SIW plan to a pddl.core.Plan, mirroring siw.PlanResult.to_pddl_plan."""
    return Plan([(action.name, [Constant(arg) for arg in action.args]) for action in plan])


def compute_plans(domain_str: str, problem_str: str, planner_config: dict[str, Any]) -> Iterator[Plan]:
    """
    Compute example plans with SIW, using the same plan syntax as topk_planner.compute_plans.

    The plans are pulled lazily from siw.solve_iter(), so a caller that stops early only pays
    for the plans it consumed. With the diverse-plans mechanisms off, SIW is deterministic and
    this yields at most one plan.
    """
    try:
        # siw.solve_iter() reads a str as a file path, so the PDDL sources are parsed here.
        stream = siw.solve_iter(
            DomainParser()(domain_str),
            ProblemParser()(problem_str),
            algorithm=planner_config["algorithm"],
            max_width=planner_config["max_width"],
            prune=planner_config["prune"],
            branch=planner_config["branch"],
            restarts=planner_config["restarts"],
            seed=planner_config["seed"],
            explore_widths=planner_config["explore_widths"],
            max_nodes=planner_config["max_nodes"],
        )
    except siw.GroundingError as e:
        log.warning(f"SIW cannot handle this task, generating no example plans: {e}")
        return
    number_of_plans = 0
    goal_holds_initially = False
    with stream:
        for plan in stream:
            if not plan:
                # An empty plan is a success, but it is useless as an example plan.
                goal_holds_initially = True
                continue
            number_of_plans += 1
            log.info(f"SIW found example plan {number_of_plans} of cost {len(plan)}")
            yield to_pddl_plan(plan)
    if not number_of_plans:
        if goal_holds_initially:
            log.info("The goal already holds initially, generating no example plans")
        else:
            log.info("SIW found no plan, generating no example plans")
