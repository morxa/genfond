import enum
import logging
import sys
from typing import Any, Collection, Iterator, Mapping, MutableMapping, Optional

from pddl.core import Plan, Problem

from .ground import state_string
from .state_space_generator import State
from .topk_planner import compute_plans

log = logging.getLogger("genfond.problem_iterator")

MAX_COST = sys.maxsize


class Result(enum.Enum):
    UNKNOWN = 0
    SUCCESS = 1
    NO_SOLUTION = 2
    OUT_OF_RESOURCES = 3


class LastStep(enum.Enum):
    START = 0
    INC_PLANS = 1
    INC_COMPLEXITY = 2


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
        self.all_features = False
        self.last_step = LastStep.START
        self.complexity = self.config["min_complexity"]
        self.last_result = Result.SUCCESS
        self.succ_complexity = self.complexity
        self.active_problems_solved = True
        self.max_cost = MAX_COST
        self.solved = {problem.name: False for problem in self.problems}
        return self

    def set_last_result(self, result: Result, cost: Optional[tuple[int]] = None) -> None:
        self.last_result = result
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
            self.active_plans.get(problem, []).append(plan)
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
            f" |selected_states|={len(self.selected_states)} states"
        )
        return {
            "active_problems": self.active_problems,
            "complexity": self.complexity,
            "all_features": self.all_features,
            "max_cost": self.max_cost,
            "example_plans": self.active_plans,
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
        return {
            "active_problems": self.active_problems,
            "complexity": self.complexity,
            "all_features": self.all_features,
            "max_cost": self.max_cost,
            "selected_states": self.selected_states,
        }
