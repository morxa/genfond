import logging
import random
from collections import deque
from collections.abc import Collection
from enum import Enum
from typing import Optional

from pddl.action import Action
from pddl.core import Domain, Plan, Problem
from pddl.logic import Predicate
from pddl.logic.base import And, Formula, Not, OneOf
from pddl.logic.effects import When
from pddl.logic.functions import (
    Assign,
    BinaryFunction,
    Decrease,
    Divide,
)
from pddl.logic.functions import EqualTo as FunctionEqualTo
from pddl.logic.functions import (
    FunctionExpression,
    GreaterEqualThan,
    GreaterThan,
    Increase,
    LesserEqualThan,
    LesserThan,
    Minus,
    NumericFunction,
    NumericValue,
    Plus,
    ScaleDown,
    ScaleUp,
    Times,
)
from pddl.logic.predicates import EqualTo

from genfond.ground import action_string, state_string

from .ground import ground
from .shutdown import stop_requested

log = logging.getLogger("genfond.state_space_generator")

type State = frozenset[Formula]

# How many nodes the expansion loop pops between two `stop_requested()` polls. Expanding a state
# grounds nothing new -- the actions are grounded once up front -- so a pop is cheap and the
# check has to be rare enough not to show up in the profile, yet frequent enough that a SIGTERM
# is seen within a fraction of a second. The flag itself is a `threading.Event`, i.e. a bare
# bool read, so this interval is conservative rather than necessary.
STOP_CHECK_INTERVAL = 128


class ExpansionInterrupted(Exception):
    """Raised out of the expansion loop when a shutdown was requested (SIGINT/SIGTERM).

    The expansion of a large unrestricted state space is the one phase of a round that can run
    for hours without ever reaching `Solver.solve`, where the shutdown flag used to be polled
    for the first time. A run killed in that phase produced nothing at all; raising here lets
    `iterative_solver` stop the way it does everywhere else.
    """


class ExpansionLimitExceeded(Exception):
    """Raised when `max_states_per_problem` is hit while expanding a problem.

    Carries the problem it happened on so the caller can defer exactly that problem for the
    round instead of losing the whole round. The graph is abandoned half-built and must not be
    used: the states still in the queue were never expanded, so `compute_alive` would call them
    alive on the strength of successors that were never generated.
    """

    def __init__(self, problem_name: str, num_states: int, max_states: int):
        super().__init__(
            f"Expanding {problem_name} exceeded max_states_per_problem={max_states} ({num_states} states)"
        )
        self.problem_name = problem_name
        self.num_states = num_states
        self.max_states = max_states


def eval_function_term(term: FunctionExpression, state: State) -> float | int:
    if isinstance(term, NumericValue):
        return term.value
    elif isinstance(term, NumericFunction):
        for f in state:
            if not isinstance(f, FunctionEqualTo):
                continue
            if f.operands[0] == term:
                return f.operands[1].value
        # TODO: check if this is correct
        # DZC: This is probably be fine. If a numeric fluent does not appear in the initial
        # state of the PDDL, its value is assumed to be 0.
        return 0
        # raise ValueError(f'Function {term} not found in state {state_string(state)}')
    elif isinstance(term, Plus):
        return eval_function_term(term.operands[0], state) + eval_function_term(term.operands[1], state)
    else:
        raise ValueError("Unknown term type: {}".format(type(term)))


def check_formula(state: State, formula: Formula) -> bool:
    if isinstance(formula, And):
        return all(check_formula(state, subformula) for subformula in formula.operands)
    elif isinstance(formula, Not):
        return not check_formula(state, formula.argument)
    elif isinstance(formula, Predicate):
        return formula in state
    elif isinstance(formula, EqualTo):
        return formula.left == formula.right
    elif isinstance(formula, LesserThan):
        return eval_function_term(formula.operands[0], state) < eval_function_term(formula.operands[1], state)
    elif isinstance(formula, LesserEqualThan):
        return eval_function_term(formula.operands[0], state) <= eval_function_term(formula.operands[1], state)
    elif isinstance(formula, GreaterThan):
        return eval_function_term(formula.operands[0], state) > eval_function_term(formula.operands[1], state)
    elif isinstance(formula, GreaterEqualThan):
        return eval_function_term(formula.operands[0], state) >= eval_function_term(formula.operands[1], state)
    elif isinstance(formula, FunctionEqualTo):
        return eval_function_term(formula.operands[0], state) == eval_function_term(formula.operands[1], state)
    else:
        raise ValueError("Unknown formula type: {}".format(type(formula)))


def apply_action_effects(state: State, action: Action) -> set[State]:
    return apply_effects(frozenset({state}), action.effect)


def _stable_successors(state: State, action: Action) -> list[State]:
    """`apply_action_effects` in a process-independent order.

    The result is a `set[State]`, and `State` is a `frozenset` of `pddl` `Predicate`s whose hash
    mixes in the identity hash of a class object (see `ground._stable_constants`), so its
    iteration order changes from process to process. That order decides which successor of a
    nondeterministic action is given the lower node id, and -- in `random_walk` -- which one a
    draw from the seeded RNG lands on. Deterministic actions have a single successor, so the
    sort is skipped for them and this stays free in the hot expansion loop.
    """
    succs = apply_action_effects(state, action)
    if len(succs) <= 1:
        return list(succs)
    return sorted(succs, key=state_string)


def apply_effects(states: Collection[State], effects: Formula | Collection[Formula]) -> set[State]:
    new_states: set[State] = set()
    for state in states:
        new_states |= apply_effects_to_state(state, effects)
        assert all(isinstance(s, Collection) for s in new_states)
        assert all(all(isinstance(f, (Predicate, FunctionEqualTo)) for f in s) for s in new_states)
    return new_states


def apply_effects_to_state(state: State, effects: Formula | Collection[Formula]) -> set[State]:
    assert all(isinstance(f, (Predicate, FunctionEqualTo)) for f in state)
    if isinstance(effects, Collection):
        states = {state}
        for effect in effects:
            states = apply_effects(states, effect)
        return states
    elif isinstance(effects, And) or isinstance(effects, And):
        states = {state}
        for effect in effects.operands:
            states = apply_effects(states, effect)
        return set(states)
    elif isinstance(effects, Predicate):
        return set({state | {effects}})
    elif isinstance(effects, Not):
        return set({frozenset([f for f in state if f != effects.argument])})
    elif isinstance(effects, When):
        if check_formula(state, effects.condition):
            return apply_effects({state}, effects.effect)
        else:
            return set({state})
    elif isinstance(effects, BinaryFunction):
        if isinstance(effects.operands[0], NumericFunction):
            fct = effects.operands[0]
            change = eval_function_term(effects.operands[1], state)
        else:
            # DZC: What is this else case?
            fct = effects.operands[1]
            change = effects.operands[0]
        assert isinstance(fct, NumericFunction)
        # DZC: remove this assert by evaluating change = eval_function_term(...)
        # assert isinstance(change, NumericValue)
        current_evals = [f for f in state if isinstance(f, FunctionEqualTo) and f.operands[0] == fct]
        if not current_evals:
            current_eval = FunctionEqualTo(fct, NumericValue(0))
        else:
            assert len(current_evals) == 1
            current_eval = current_evals[0]
        current_value = current_eval.operands[1].value
        if isinstance(effects, Assign):
            new_value = change
        elif isinstance(effects, Increase):
            new_value = current_value + change
        elif isinstance(effects, Decrease):
            new_value = current_value - change
        elif isinstance(effects, (Plus, Minus, Times, Divide, ScaleUp, ScaleDown)):
            raise NotImplementedError()
        else:
            raise ValueError("Unknown effect type: {}".format(type(effects)))
        # log.debug(f'Change {fct} from {current_value} to {new_value}')
        return frozenset(
            {frozenset([f for f in state if f != current_eval] + [FunctionEqualTo(fct, NumericValue(new_value))])}
        )
    elif isinstance(effects, OneOf):
        new_states = set()
        for effect in effects.operands:
            new_states |= apply_effects({state}, effect)
        return frozenset(new_states)
    else:
        raise ValueError("Unknown effect type: {}".format(type(effects)))


class Alive(Enum):
    ALIVE = 0
    DEAD = 1
    UNKNOWN = 2
    PRUNED = 3
    NUM_PRUNED = 4


class StateSpaceNode:

    def __init__(self, state: State, id: int, plan_suffixes: Optional[list] = None):
        self.state = state
        self.id = id
        self.plan_suffixes: list = []
        self._plan_suffix_keys: set[tuple] = set()
        self.children: dict[Action, set[StateSpaceNode]] = dict()
        self.alive = Alive.UNKNOWN
        self.goal = False
        self.parents: set[StateSpaceNode] = set()
        if plan_suffixes:
            self.add_plan_suffixes(plan_suffixes)

    def add_plan_suffixes(self, plan_suffixes: Collection) -> bool:
        """Register plan continuations, ignoring ones already known.

        Returns whether anything was actually added. A node can be reached again long after
        it was expanded, so the caller has to re-expand it when this reports new suffixes --
        otherwise the actions those suffixes prescribe here are never matched.
        """
        added = False
        for suffix in plan_suffixes:
            key = tuple(suffix)
            if key in self._plan_suffix_keys:
                continue
            self._plan_suffix_keys.add(key)
            self.plan_suffixes.append(suffix)
            added = True
        return added

    def __str__(self) -> str:
        return state_string(self.state)

    def __repr__(self) -> str:
        return repr(self.state)

    def add_child(self, action: Action, node: "StateSpaceNode") -> None:
        self.children.setdefault(action, set()).add(node)


def get_num_vals(state: State) -> set[int]:
    return {f.operands[1].value for f in state if isinstance(f, FunctionEqualTo)}


def plan_string(plan):
    return " ".join([action_string(action) for action in plan])


def plan_visited_states(domain: Domain, problem: Problem, plan: Plan) -> set[State]:
    """The states `plan` reaches when replayed from `problem.init`, without expanding the graph.

    Mirrors the plan-matching branch of `StateSpaceGraph.__init__`: at a nondeterministic action
    the actual runtime outcome is unknown, so every effect outcome is counted as visited, exactly
    as a plan suffix there is propagated to *all* matching successors, not just one. A step whose
    precondition fails along some branch simply drops that branch (no successor states from it),
    which cannot happen for a genuinely valid plan but is handled defensively all the same.

    This only follows the plan's own actions -- it does not ground or check every action of the
    domain -- so it is far cheaper than building (or extending) a `StateSpaceGraph`. Used to
    dedupe candidate example plans by the states they would actually add.
    """
    actions = plan.instantiate(domain)
    frontier = {problem.init}
    visited = {problem.init}
    for action in actions:
        successors: set[State] = set()
        for state in frontier:
            if not check_formula(state, action.precondition):
                continue
            successors |= apply_action_effects(state, action)
        frontier = successors
        visited |= frontier
    return visited


class StateSpaceGraph:

    def __init__(
        self,
        domain: Domain,
        problem: Problem,
        prune: bool = True,
        selected_states: Optional[set[State]] = None,
        max_num_val: Optional[int] = None,
        plans: Optional[list[Plan]] = None,
        frontier: bool = False,
        dead_states: Optional[Collection[State]] = None,
        max_states: Optional[int] = None,
    ):
        self.domain = domain
        self.problem = problem
        self.next_id = 0
        if plans:
            # Instantiate the actions in the plan, as they are not Action objects yet.
            plans = [plan.instantiate(domain) for plan in plans]
        else:
            plans = []
        queue = []
        self.nodes: dict[State, StateSpaceNode] = dict()
        if selected_states:
            for state in selected_states:
                node = StateSpaceNode(state, self.next_id)
                self.next_id += 1
                self.nodes[state] = node
                queue.append(node)
        else:
            root_state = problem.init
            self.root = StateSpaceNode(root_state, 0, plans)
            self.next_id = 1
            self.nodes = {root_state: self.root}
            queue = [self.root]
        grounded_actions = ground(domain, problem)
        pops = 0
        while queue:
            pops += 1
            if pops % STOP_CHECK_INTERVAL == 0 and stop_requested():
                # Always on, independent of max_states: without it a SIGTERM delivered during a
                # multi-hour unrestricted expansion is only seen once that expansion finishes on
                # its own, which is precisely the case it exists for.
                raise ExpansionInterrupted(f"Expansion of {problem.name} interrupted by a stop request")
            if max_states is not None and len(self.nodes) >= max_states:
                # Checked per popped node rather than per generated successor, so the final count
                # may overshoot by one node's worth of successors. That is deliberate: the limit
                # is a safety net against an expansion that would never finish, not an exact
                # budget, and one branching factor of slack costs nothing.
                log.warning(
                    "Expanding %s reached max_states_per_problem=%d with %d state(s) still queued;"
                    " abandoning the expansion",
                    problem.name,
                    max_states,
                    len(queue),
                )
                raise ExpansionLimitExceeded(problem.name, len(self.nodes), max_states)
            node = queue.pop()
            state = node.state
            if check_formula(state, problem.goal):
                node.alive = Alive.ALIVE
                node.goal = True
            for action in grounded_actions:
                if not check_formula(state, action.precondition):
                    continue
                for succ in _stable_successors(node.state, action):
                    plan_suffixes = [plan[1:] for plan in node.plan_suffixes if plan and plan[0] == action]
                    matches_plan = bool(plan_suffixes)
                    log.debug(
                        "%s -- %s -> %s has %d matching plans, plan_suffixes: %s",
                        state_string(state),
                        action_string(action),
                        state_string(succ),
                        len(plan_suffixes),
                        [plan_string(plan) for plan in plan_suffixes],
                    )
                    new, new_node, gained_suffixes = self.add_node(succ, state, action, plan_suffixes)
                    known_dead = dead_states is not None and succ in dead_states
                    if new:
                        if max_num_val and any(v > max_num_val for v in get_num_vals(succ)):
                            new_node.alive = Alive.NUM_PRUNED
                        elif known_dead:
                            # Refuted earlier in this run: never expand it, and emit no
                            # pruned/2 fact, so the solver cannot select a transition into it.
                            log.debug(f"Marking known dead end {state_string(succ)}")
                            new_node.alive = Alive.DEAD
                        elif selected_states and succ not in selected_states:
                            log.debug(f"Pruning {state_string(succ)}")
                            new_node.alive = Alive.PRUNED
                        elif plans and not matches_plan:
                            if check_formula(succ, problem.goal):
                                # An off-plan successor that already satisfies the goal. It is
                                # never queued, so the goal check in the main loop would never
                                # run on it and it would look like a dead end (or, with
                                # frontier expansion on, get handed to the planner).
                                new_node.alive = Alive.ALIVE
                                new_node.goal = True
                            elif frontier:
                                new_node.alive = Alive.PRUNED
                            else:
                                new_node.alive = Alive.DEAD
                        else:
                            queue.append(new_node)
                    revivable = (Alive.DEAD,) if selected_states else (Alive.DEAD, Alive.PRUNED)
                    if plans and matches_plan and new_node.alive in revivable:
                        if known_dead:
                            log.warning(
                                "A plan runs through %s, which was refuted as a dead end earlier;"
                                " trusting the plan. This means the planner contradicted itself.",
                                state_string(succ),
                            )
                        log.debug(f"Reviving {state_string(succ)} because it matches a plan")
                        queue.append(new_node)
                        new_node.alive = Alive.UNKNOWN
                    elif gained_suffixes and not known_dead:
                        # The node was reached again by a plan it did not know about. It may
                        # already have been expanded, in which case the actions this new
                        # suffix prescribes here were never matched and its successors were
                        # left off-plan, so it has to be expanded again. This terminates
                        # because a node only re-enters the queue when its suffix set grows,
                        # and that set is finite.
                        log.debug(f"Re-expanding {state_string(succ)}, it gained a plan suffix")
                        queue.append(new_node)
        compute_alive(self.nodes.values())
        if prune:
            self.prune_nodes()
        assert all(node.alive != Alive.UNKNOWN for node in self.nodes.values())
        # assert self.root.alive == Alive.ALIVE, 'Problem {} is unsolvable'.format(problem.name)

    def add_node(
        self, state: State, parent_state: State, action: Action, plan_suffixes: list[list[Action]]
    ) -> tuple[bool, StateSpaceNode, bool]:
        parent = self.nodes[parent_state]
        try:
            node = self.nodes[state]
            gained_suffixes = node.add_plan_suffixes(plan_suffixes)
            new = False
        except KeyError:
            node = StateSpaceNode(state, self.next_id, plan_suffixes)
            self.next_id += 1
            self.nodes[state] = node
            new = True
            gained_suffixes = False

        parent.add_child(action, node)
        node.parents.add(parent)
        return new, node, gained_suffixes

    def action_path_from_root(self, node: "StateSpaceNode") -> Optional[list[Action]]:
        """Return a shortest action sequence leading from the root to `node`.

        Returns None if this graph has no root (the `selected_states` mode) or `node` is
        unreachable. The sequence is only a plan for deterministic domains; for an action
        with several outcomes, replaying it may end up somewhere else.
        """
        root = getattr(self, "root", None)
        if root is None:
            return None
        if node is root:
            return []
        came_from: dict[StateSpaceNode, tuple[StateSpaceNode, Action]] = dict()
        seen = {root}
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for action, children in current.children.items():
                if len(children) > 1:
                    log.debug(
                        "Action %s in %s is nondeterministic; the action path is not a plan",
                        action_string(action),
                        self.problem.name,
                    )
                for child in children - seen:
                    seen.add(child)
                    came_from[child] = (current, action)
                    if child is node:
                        path = []
                        step = node
                        while step in came_from:
                            step, edge = came_from[step]
                            path.append(edge)
                        return list(reversed(path))
                    queue.append(child)
        return None

    def prune_nodes(self) -> None:
        pruned_dead = []
        pruned_selected: list[State] = []
        for state, node in self.nodes.items():
            if node.alive == Alive.DEAD and all([parent.alive == Alive.DEAD for parent in node.parents]):
                pruned_dead.append(state)
        before = len(self.nodes)
        for state in pruned_dead + pruned_selected:
            del self.nodes[state]
        for node in self.nodes.values():
            for action, children in node.children.items():
                node.children[action] = {child for child in children if child.state in self.nodes}
        log.info(f"Pruned {len(pruned_dead)} dead " f"out of {before} states in {self.problem.name}")


def generate_state_space(domain: Domain, problem: Problem, *args, **kwargs):
    return StateSpaceGraph(domain, problem, *args, **kwargs)


def can_reach(node: StateSpaceNode, goal_nodes: Collection[StateSpaceNode]) -> bool:
    seen = set()
    stack = [node]
    while stack:
        current_node = stack.pop()
        if current_node in goal_nodes:
            return True
        if current_node.alive == Alive.DEAD:
            continue
        if current_node.alive == Alive.ALIVE:
            return True
        if current_node in seen:
            continue
        seen.add(current_node)
        for children in current_node.children.values():
            if all(child.alive != Alive.DEAD for child in children):
                stack.extend(children)
    return False


def find_nodes_leading_to_dead(nodes: Collection[StateSpaceNode]) -> bool:
    queue = [node for node in nodes if node.alive == Alive.UNKNOWN]
    changed = False
    while queue:
        node = queue.pop()
        if all(any(child.alive == Alive.DEAD for child in children) for children in node.children.values()):
            node.alive = Alive.DEAD
            changed = True
            for parent in node.parents:
                if parent.alive == Alive.UNKNOWN:
                    queue.append(parent)
    return changed


def find_node_not_reaching_goal(nodes: Collection[StateSpaceNode]) -> bool:
    queue = [node for node in nodes if node.alive == Alive.UNKNOWN]
    goal_nodes = [node for node in nodes if node.alive in [Alive.ALIVE, Alive.PRUNED]]
    changed = False
    while queue:
        node = queue.pop()
        if not can_reach(node, goal_nodes):
            node.alive = Alive.DEAD
            changed = True
            for parent in node.parents:
                if parent.alive == Alive.UNKNOWN:
                    queue.append(parent)
    return changed


def compute_alive(nodes: Collection[StateSpaceNode]) -> None:
    changed = True
    while changed:
        changed = find_nodes_leading_to_dead(nodes)
        changed = find_node_not_reaching_goal(nodes) or changed
    for node in nodes:
        if node.alive == Alive.UNKNOWN:
            node.alive = Alive.ALIVE


def random_walk(domain: Domain, problem: Problem, initial_states: set[State], max_steps: int = 100):
    states = sorted(initial_states, key=state_string)
    grounded_actions = ground(domain, problem)
    while True:
        state = random.choice(states)
        for _ in range(max_steps):
            if check_formula(state, problem.goal):
                return states
            applicable_actions = [action for action in grounded_actions if check_formula(state, action.precondition)]
            if not applicable_actions:
                break
            action = random.choice(applicable_actions)
            succ = random.choice(_stable_successors(state, action))
            states.append(succ)
            state = succ
            state = succ
