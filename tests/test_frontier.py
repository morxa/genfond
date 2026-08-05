from pddl.parser.plan import PlanParser
from pddl.parser.problem import ProblemParser

from genfond.frontier import _root_anchored_plan, reroot
from genfond.state_space_generator import Alive, generate_state_space


def test_reroot_round_trips_through_the_parser(typed_blocks_medsize):
    domain, problem = typed_blocks_medsize
    state_space = generate_state_space(domain, problem)
    node = next(n for n in state_space.nodes.values() if n.state != problem.init)

    rerooted = reroot(problem, node.state, str(node.id))
    assert set(rerooted.init) == set(node.state)
    assert rerooted.goal == problem.goal
    assert rerooted.objects == problem.objects

    # The planner receives the problem as a PDDL string, so it has to survive a round trip
    # including the typed object declarations.
    parsed = ProblemParser()(str(rerooted))
    assert set(parsed.init) == set(node.state)
    assert parsed.objects == problem.objects


def test_root_anchored_plan_reaches_the_frontier_state(typed_blocks_medsize):
    domain, problem = typed_blocks_medsize
    plan = PlanParser()("(pick a table) (put a b)")
    state_space = generate_state_space(domain, problem, plans=[plan], frontier=True)
    frontier_node = next(n for n in state_space.nodes.values() if n.alive == Alive.PRUNED)
    prefix = state_space.action_path_from_root(frontier_node)
    assert prefix

    # A plan found from the frontier state has to be spliced onto the path that reaches it,
    # because StateSpaceGraph always replays plans from the problem's initial state.
    suffix = PlanParser()("(put a table)")
    anchored = _root_anchored_plan(prefix, suffix)
    assert len(anchored.actions) == len(prefix) + 1
    assert anchored.instantiate(domain)[: len(prefix)] == list(prefix)

    # Feeding it back must make the previously unexpanded state part of the explored space.
    expanded = generate_state_space(domain, problem, plans=[plan, anchored], frontier=True)
    assert expanded.nodes[frontier_node.state].alive != Alive.PRUNED
    assert expanded.nodes[frontier_node.state].children
