import enum
import logging
import sys
from typing import Any, Collection, Iterator, Mapping, MutableMapping, Optional

from pddl.core import Plan, Problem

from .ground import state_string
from .state_space_generator import State

log = logging.getLogger("genfond.problem_iterator")

MAX_COST = sys.maxsize


def plan_key(plan: Plan) -> tuple:
    """A hashable identity for a plan. `Plan` defines __eq__ but no __hash__."""
    return tuple((str(name), tuple(str(arg) for arg in args)) for name, args in plan.actions)


class Result(enum.Enum):
    UNKNOWN = 0
    SUCCESS = 1
    NO_SOLUTION = 2
    OUT_OF_RESOURCES = 3
    # A model was found, but it relies on transitions into unexpanded frontier states, so it
    # is not a policy yet. The caller expands those states and the same configuration is
    # retried with the enlarged plan set.
    FRONTIER = 4


class LastStep(enum.Enum):
    START = 0
    INC_PLANS = 1
    INC_COMPLEXITY = 2
    EXPAND_FRONTIER = 3


class ProblemIterator:

    def __init__(self, problems: list[Problem], config: Mapping, plans: Optional[Mapping[str, Iterator[Plan]]] = None):
        self.problems = problems
        self.config = config
        self.plan_iterators = plans

    def __iter__(self) -> "ProblemIterator":
        self.active_problems: list[Problem] = []
        self.active_plans: MutableMapping[str, Plan] = dict()
        self.selected_states: dict[str, set[State]] = dict()
        self.new_states: dict[str, set[State]] = dict()
        self.dead_states: dict[str, set[State]] = dict()
        self.frontier_expansions = 0
        self.frontier_progress = False
        self.all_features = False
        self.last_step = LastStep.START
        self.complexity = self.config["min_complexity"]
        self.last_result = Result.SUCCESS
        self.succ_complexity = self.complexity
        self.active_problems_solved = True
        self.max_cost = MAX_COST
        self.solved = {problem.name: False for problem in self.problems}
        return self

    def record_frontier_expansion(
        self, plans: Mapping[str, list[Plan]], dead_states: Mapping[str, set[State]]
    ) -> None:
        """Take the result of expanding the frontier states the last model relied on.

        Sets `frontier_progress`, which gates the retry: without new plans and without new
        dead ends the next round would be identical, so we must fall through to the normal
        escalation ladder instead of looping forever.
        """
        self.frontier_expansions += 1
        self.frontier_progress = False
        for problem_name, new_plans in plans.items():
            active = self.active_plans.setdefault(problem_name, [])
            known = {plan_key(plan) for plan in active}
            for plan in new_plans:
                if plan_key(plan) in known:
                    # The planner is deterministic, so a frontier state that the new plan
                    # fails to expand yields the same plan every round. Re-adding it is not
                    # progress; counting it as such spins this loop until the expansion
                    # budget runs out.
                    log.debug("Frontier plan for %s is already an example plan, ignoring it", problem_name)
                    continue
                known.add(plan_key(plan))
                active.append(plan)
                self.frontier_progress = True
        for problem_name, states in dead_states.items():
            known_dead = self.dead_states.setdefault(problem_name, set())
            if states - known_dead:
                self.frontier_progress = True
            known_dead |= states

    def set_last_result(self, result: Result, cost: Optional[tuple[int]] = None) -> None:
        self.last_result = result
        # A frontier model's feature cost is artificially low because it was allowed to assume
        # unexpanded states are solvable. Tightening max_cost or marking problems solved from
        # it would make every later round unsatisfiable, so Result.FRONTIER changes nothing.
        if result == Result.SUCCESS:
            assert cost
            self.active_problems_solved = True
            self.max_cost = cost[-1] - 1
            self.succ_complexity = self.complexity
            # self.solved = {
            #     problem.name: True if problem in self.active_problems else False
            #     for problem in self.problems
            # }
            self.new_states.clear()
            for problem in self.problems:
                if problem.name in self.selected_states and problem not in self.active_problems:
                    del self.selected_states[problem.name]

    def set_solved(self, problem: Problem, solved: bool = True):
        self.solved[problem.name] = solved

    def get_unsolved_problems(self) -> list[Problem]:
        return [problem for problem in self.problems if not self.solved[problem.name]]

    def __next__(self) -> Mapping[str, Any]:
        assert self.last_result != Result.UNKNOWN, "You must set the result of the last problem before calling next"
        log.debug(
            f"last result: {self.last_result.name}, all features: {self.all_features}, complexity: {self.complexity}"
        )
        if (
            self.last_result == Result.FRONTIER
            and self.frontier_progress
            and self.frontier_expansions <= self.config["max_frontier_expansions"]
        ):
            # Retry the exact same configuration; only the plan and dead-end sets grew.
            self.last_step = LastStep.EXPAND_FRONTIER
        elif (
            (self.last_step == LastStep.INC_COMPLEXITY or self.complexity == self.config["max_complexity"])
            and self.active_problems
            and not self.active_problems_solved
            and self.last_result != Result.OUT_OF_RESOURCES
            and self.plan_iterators
            and (
                # Find the next plan for an active problem that is not yet solved
                found := next(
                    (
                        (k.name, v)
                        for k in self.active_problems
                        if not self.solved[k.name] and (v := next(self.plan_iterators[k.name], None)) is not None
                    ),
                    None,
                )
            )
        ):
            problem, plan = found
            self.active_plans.setdefault(problem, []).append(plan)
            self.all_features = False
            self.max_cost = MAX_COST
            # self.complexity = self.succ_complexity
            self.last_step = LastStep.INC_PLANS
        elif (
            self.active_problems
            and self.last_result != Result.OUT_OF_RESOURCES
            and not self.all_features
            and self.config["use_unrestricted_features"]
        ):
            self.all_features = True
        elif (
            self.active_problems
            and self.last_result != Result.OUT_OF_RESOURCES
            and (self.all_features or not self.config["use_unrestricted_features"])
            and self.complexity < self.config["max_complexity"]
            and self.max_cost > self.complexity
        ):
            self.all_features = False
            self.complexity += 1
            self.last_step = LastStep.INC_COMPLEXITY
        elif self.active_problems_solved and any(not solved for solved in self.solved.values()):
            self.all_features = False
            self.max_cost = MAX_COST
            self.active_problems_solved = False
            self.complexity = self.succ_complexity
            self.last_step = LastStep.START
            next_problem = next(
                problem
                for problem in self.problems
                if not self.solved[problem.name] and problem not in self.active_problems
            )
            if (
                self.config["unselect_problems"]
                and self.active_problems
                and self.problems.index(next_problem)
                > max([self.problems.index(problem) for problem in self.active_problems])
            ):
                self.active_problems = [next_problem]
            else:
                self.active_problems.append(next_problem)
            if self.plan_iterators:
                self.active_plans[next_problem.name] = []
                while len(self.active_plans[next_problem.name]) < self.config["min_number_of_plans"]:
                    next_plan = next(self.plan_iterators[next_problem.name], None)
                    if next_plan is None:
                        break
                    self.active_plans[next_problem.name].append(next_plan)
        else:
            raise StopIteration
        log.debug(
            f'Next set: {", ".join([p.name for p in self.active_problems])},'
            f" complexity={self.complexity},"
            f" all_features={self.all_features},"
            f' max_cost={self.max_cost if self.max_cost < MAX_COST else "MAX_COST"},'
            f" |selected_states|={len(self.selected_states)} states,"
            f" last_step={self.last_step.name}"
        )
        return {
            "active_problems": self.active_problems,
            "complexity": self.complexity,
            "all_features": self.all_features,
            "max_cost": self.max_cost,
            "example_plans": self.active_plans,
            "dead_states": self.dead_states,
        }


class OneShotProblemIterator(ProblemIterator):

    def __iter__(self, *args, **kwargs) -> "OneShotProblemIterator":
        super().__iter__(*args, **kwargs)
        self.active_problems = list(self.problems)
        self.called = False
        self.all_features = self.config["use_unrestricted_features"]
        self.complexity = self.config["max_complexity"]
        if self.config["use_selected_states"]:
            self.new_states = {problem.name: {problem.init} for problem in self.active_problems}
        return self

    def __next__(self) -> Mapping[str, Any]:
        if self.called:
            raise StopIteration
        self.called = True
        # Must match ProblemIterator.__next__: solve_step takes example_plans, not
        # selected_states, so returning the latter used to make --one-shot raise TypeError.
        return {
            "active_problems": self.active_problems,
            "complexity": self.complexity,
            "all_features": self.all_features,
            "max_cost": self.max_cost,
            "example_plans": self.active_plans,
            "dead_states": self.dead_states,
        }
