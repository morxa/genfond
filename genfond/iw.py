import itertools
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Collection, Iterable, Iterator

import pymimir
from pddl.parser.plan import Plan, PlanParser

PathLike = str | Path
StateCallback = Callable[[pymimir.State], None]
TransitionCallback = Callable[[pymimir.State, pymimir.GroundAction, float, pymimir.State], None]
_MAX_COMPUTE_PLANS_WIDTH = 1


@dataclass(frozen=True, slots=True)
class ParsedProblem:
    domain: pymimir.Domain
    problem: pymimir.Problem
    initial_state: pymimir.State


@dataclass(slots=True)
class _SearchNode:
    state: pymimir.State
    actions: tuple[pymimir.GroundAction, ...]
    depth: int


def _as_existing_file(path: PathLike, description: str) -> Path:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} file not found: {file_path}")
    return file_path


def set_compute_plans_max_width(width: int = 1) -> None:
    global _MAX_COMPUTE_PLANS_WIDTH
    if width <= 0:
        raise ValueError("width must be positive")
    _MAX_COMPUTE_PLANS_WIDTH = width


def get_compute_plans_max_width() -> int:
    return _MAX_COMPUTE_PLANS_WIDTH


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


def _action_to_lisp_str(action: pymimir.GroundAction) -> str:
    action_name = action.get_action().get_name()
    action_args = [obj.get_name() for obj in action.get_objects()]
    return f"({action_name} {' '.join(action_args)})"


def _as_ordered_atom_tuple(atom_tuple: Iterable[pymimir.GroundAtom]) -> tuple[pymimir.GroundAtom, ...]:
    return tuple(sorted(atom_tuple, key=lambda atom: atom.get_index()))


class _NoveltyTracker:
    def __init__(self, width: int):
        self.width = width
        self.known_tuples: set[tuple[pymimir.GroundAtom, ...]] = set()

    def _tuple_generator(self, atoms: Collection[pymimir.GroundAtom]) -> Iterator[tuple[pymimir.GroundAtom, ...]]:
        for arity in range(1, self.width + 1):
            match arity:
                case 1:
                    for atom in atoms:
                        yield _as_ordered_atom_tuple((atom,))
                case 2:
                    for atom1 in atoms:
                        for atom2 in atoms:
                            yield _as_ordered_atom_tuple((atom1, atom2))
                case _:
                    for atom_tuple in itertools.combinations(atoms, arity):
                        yield _as_ordered_atom_tuple(atom_tuple)

    def observe(self, atoms: Collection[pymimir.GroundAtom]) -> bool:
        is_novel = False
        for atom_tuple in self._tuple_generator(atoms):
            if atom_tuple in self.known_tuples:
                continue
            self.known_tuples.add(atom_tuple)
            is_novel = True
        return is_novel


def _is_goal_state(state: pymimir.State) -> bool:
    return all(state.literal_holds(literal) for literal in state.get_problem().get_goal_condition())


def _positive_effect_atoms(action: pymimir.GroundAction) -> list[pymimir.GroundAtom]:
    return sum((cond_effect.get_effect().get_add_list() for cond_effect in action.get_conditional_effect()), [])


def _enumerate_iw_action_sequences(
    start_state: pymimir.State,
    width: int,
    *,
    depth_1_is_novel: bool = True,
) -> Iterator[tuple[pymimir.GroundAction, ...]]:
    novelty = _NoveltyTracker(width)
    novelty.observe(start_state.get_atoms(ignore_static=True))
    visit_queue: deque[_SearchNode] = deque([_SearchNode(start_state, tuple(), 0)])

    while visit_queue:
        node = visit_queue.popleft()
        for action in node.state.generate_applicable_actions():
            child_state = action.apply(node.state)
            pos_effect_atoms = _positive_effect_atoms(action)
            if not pos_effect_atoms:
                continue

            is_novel = novelty.observe(pos_effect_atoms)
            if node.depth == 0 and depth_1_is_novel and child_state != node.state:
                is_novel = True

            child_actions = node.actions + (action,)
            is_goal = _is_goal_state(child_state)
            if is_goal:
                yield child_actions
                continue
            if is_novel:
                visit_queue.append(_SearchNode(child_state, child_actions, node.depth + 1))


def compute_plans(domain_str: str, problem_str: str, number_of_plans: int = 3) -> Iterator[Plan]:
    """
    Compute plans with IW using the same interface and plan syntax as topk_planner.compute_plans.
    """
    if number_of_plans <= 0:
        return

    plan_parser = PlanParser()
    yielded_plans = 0
    seen_plan_strings: set[str] = set()
    max_width = get_compute_plans_max_width()

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        domain_path = temp_path / "domain.pddl"
        problem_path = temp_path / "problem.pddl"
        domain_path.write_text(domain_str)
        problem_path.write_text(problem_str)

        for mode in ("lifted", "grounded"):
            parsed_problem = load_problem(domain_path, problem_path, mode=mode)
            for width in range(1, max_width + 1):
                for action_sequence in _enumerate_iw_action_sequences(parsed_problem.initial_state, width):
                    plan_str = " ".join(_action_to_lisp_str(action) for action in action_sequence)
                    if plan_str in seen_plan_strings:
                        continue

                    seen_plan_strings.add(plan_str)
                    yield plan_parser(plan_str)
                    yielded_plans += 1

                    if yielded_plans >= number_of_plans:
                        return


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


__all__ = [
    "IWPlanner",
    "ParsedProblem",
    "compute_plans",
    "get_compute_plans_max_width",
    "load_problem",
    "set_compute_plans_max_width",
    "solve_iw",
]
