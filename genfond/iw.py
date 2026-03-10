from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pymimir

PathLike = str | Path
StateCallback = Callable[[pymimir.State], None]
TransitionCallback = Callable[[pymimir.State, pymimir.GroundAction, float, pymimir.State], None]


@dataclass(frozen=True, slots=True)
class ParsedProblem:
    domain: pymimir.Domain
    problem: pymimir.Problem
    initial_state: pymimir.State


def _as_existing_file(path: PathLike, description: str) -> Path:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} file not found: {file_path}")
    return file_path


def load_problem(
    domain_path: PathLike,
    problem_path: PathLike,
    *,
    mode: str = "lifted",
) -> ParsedProblem:
    domain_file = _as_existing_file(domain_path, "Domain")
    problem_file = _as_existing_file(problem_path, "Problem")
    domain = pymimir.Domain(domain_file)
    problem = pymimir.Problem(domain, problem_file, mode=mode)
    return ParsedProblem(domain=domain, problem=problem, initial_state=problem.get_initial_state())


class IWPlanner:
    def __init__(self, width: int, *, mode: str = "lifted"):
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width
        self.mode = mode

    def solve(
        self,
        domain_path: PathLike,
        problem_path: PathLike,
        *,
        start_state: pymimir.State | None = None,
        on_expand_state: StateCallback | None = None,
        on_expand_goal_state: StateCallback | None = None,
        on_generate_state: TransitionCallback | None = None,
        on_generate_new_state: TransitionCallback | None = None,
        on_prune_state: TransitionCallback | None = None,
    ) -> pymimir.SearchResult:
        parsed_problem = load_problem(domain_path, problem_path, mode=self.mode)
        if start_state is None:
            start_state = parsed_problem.initial_state
        elif start_state.get_problem().get_index() != parsed_problem.problem.get_index():
            raise ValueError("start_state must belong to the parsed problem")

        return pymimir.iw(
            parsed_problem.problem,
            start_state,
            self.width,
            on_expand_state=on_expand_state,
            on_expand_goal_state=on_expand_goal_state,
            on_generate_state=on_generate_state,
            on_generate_new_state=on_generate_new_state,
            on_prune_state=on_prune_state,
        )


def solve_iw(
    domain_path: PathLike,
    problem_path: PathLike,
    width: int,
    *,
    mode: str = "lifted",
    start_state: pymimir.State | None = None,
    on_expand_state: StateCallback | None = None,
    on_expand_goal_state: StateCallback | None = None,
    on_generate_state: TransitionCallback | None = None,
    on_generate_new_state: TransitionCallback | None = None,
    on_prune_state: TransitionCallback | None = None,
) -> pymimir.SearchResult:
    planner = IWPlanner(width, mode=mode)
    return planner.solve(
        domain_path,
        problem_path,
        start_state=start_state,
        on_expand_state=on_expand_state,
        on_expand_goal_state=on_expand_goal_state,
        on_generate_state=on_generate_state,
        on_generate_new_state=on_generate_new_state,
        on_prune_state=on_prune_state,
    )


__all__ = ["IWPlanner", "ParsedProblem", "load_problem", "solve_iw"]
