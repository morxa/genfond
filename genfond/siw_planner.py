import logging
from typing import Any, Iterator

import siw
from pddl.core import Plan
from pddl.parser.domain import DomainParser
from pddl.parser.problem import ProblemParser

log = logging.getLogger("genfond.siw_planner")


def compute_plans(domain_str: str, problem_str: str, planner_config: dict[str, Any]) -> Iterator[Plan]:
    """
    Compute a single example plan with SIW, using the same plan syntax as topk_planner.compute_plans.

    SIW is deterministic and returns one plan, so this yields at most one plan.
    """
    try:
        # siw.solve() reads a str as a file path, so the PDDL sources are parsed here.
        result = siw.solve(
            DomainParser()(domain_str),
            ProblemParser()(problem_str),
            algorithm=planner_config["algorithm"],
            max_width=planner_config["max_width"],
            prune=planner_config["prune"],
        )
    except siw.GroundingError as e:
        log.warning(f"SIW cannot handle this task, generating no example plans: {e}")
        return
    if not result.solved:
        log.info("SIW found no plan, generating no example plans")
        return
    if not result.plan:
        log.info("The goal already holds initially, generating no example plans")
        return
    log.info(f"SIW found a plan of cost {result.cost}, effective widths {result.effective_widths}")
    yield result.to_pddl_plan()
