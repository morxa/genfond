from pddl.logic import Constant, Predicate, constants
from pddl.parser.plan import PlanParser

from genfond.ground import ground
from genfond.state_space_generator import (
    Alive,
    apply_action_effects,
    check_formula,
    generate_state_space,
)

from .helpers import get_action


def test_generate_state_space_simple_blocks(simple_blocks):
    domain, problem = simple_blocks
    ground_actions = ground(domain, problem)
    state_space = generate_state_space(domain, problem)
    assert state_space.root.state == problem.init
    a, b, c = constants("a b c")
    pick_ab = get_action(ground_actions, "pick", (a, b))
    assert len(state_space.nodes) == 4
    assert set(state_space.root.children.keys()) == {pick_ab}
    assert len(state_space.root.children[pick_ab]) == 1
    assert next(iter(state_space.root.children[pick_ab])).state == {Predicate("holding", a)}
    put_ab = get_action(ground_actions, "put", (a, b))
    put_aa = get_action(ground_actions, "put", (a, a))
    put_ac = get_action(ground_actions, "put", (a, c))
    pick_ab_node = next(iter(state_space.root.children[pick_ab]))
    assert set(pick_ab_node.children.keys()) == {put_ab, put_aa, put_ac}
    assert len(pick_ab_node.children[put_ab]) == 1
    put_ab_node = next(iter(pick_ab_node.children[put_ab]))
    assert put_ab_node.state == {Predicate("on", a, b)}
    assert len(pick_ab_node.children[put_aa]) == 1
    put_aa_node = next(iter(pick_ab_node.children[put_aa]))
    assert put_aa_node.state == {Predicate("on", a, a)}


def test_generate_state_space_fond_blocks(fond_blocks):
    domain, problem = fond_blocks
    ground_actions = ground(domain, problem)
    state_space = generate_state_space(domain, problem)
    assert state_space.root.state == problem.init
    a, b, c, table = constants("A B C Table")
    cta_puton = get_action(ground_actions, "puton", (c, table, a))
    cba_puton = get_action(ground_actions, "puton", (c, b, a))
    bct_puton = get_action(ground_actions, "puton", (b, c, table))
    assert set(state_space.root.children.keys()) == {cta_puton, cba_puton, bct_puton}
    assert len(state_space.root.children[cta_puton]) == 1
    assert {node.state for node in state_space.root.children[cta_puton]} == apply_action_effects(
        problem.init, cta_puton
    )
    assert len(state_space.root.children[cba_puton]) == 2
    # cta_puton and cba_puton lead to a common node
    assert len(state_space.root.children[cta_puton] & state_space.root.children[cba_puton]) == 1
    assert {node.state for node in state_space.root.children[cba_puton]} == apply_action_effects(
        problem.init, cba_puton
    )
    assert len(state_space.root.children[bct_puton]) == 2
    assert {node.state for node in state_space.root.children[bct_puton]} == apply_action_effects(
        problem.init, bct_puton
    )
    cta_puton_node = next(iter(state_space.root.children[cta_puton]))
    abt_puton = get_action(ground_actions, "puton", (a, b, table))
    act_puton = get_action(ground_actions, "puton", (a, c, table))
    bat_puton = get_action(ground_actions, "puton", (b, a, table))
    bct_puton = get_action(ground_actions, "puton", (b, c, table))
    cat_puton = get_action(ground_actions, "puton", (c, a, table))
    cbt_puton = get_action(ground_actions, "puton", (c, b, table))
    assert set(cta_puton_node.children.keys()) == {abt_puton, act_puton, bat_puton, bct_puton, cat_puton, cbt_puton}
    assert cta_puton_node.children[cat_puton] == {state_space.root, cta_puton_node}


def test_plan_input_simple_blocks(typed_blocks_medsize):
    domain, problem = typed_blocks_medsize
    ground_actions = ground(domain, problem)
    a, b, c, d = constants("a b c d", "block")
    table = Constant("table", "obj")
    plan = PlanParser()("(pick a table) (put a b)")
    state_space = generate_state_space(domain, problem, plans=[plan])
    assert state_space.root.state == problem.init
    assert len(state_space.root.children) == 4
    pickatable = get_action(ground_actions, "pick", (a, table))
    pickbtable = get_action(ground_actions, "pick", (b, table))
    pickctable = get_action(ground_actions, "pick", (c, table))
    pickdtable = get_action(ground_actions, "pick", (d, table))
    assert set(state_space.root.children.keys()) == {pickatable, pickbtable, pickctable, pickdtable}
    s_picka = next(iter(state_space.root.children[pickatable]))
    s_pickb = next(iter(state_space.root.children[pickbtable]))
    s_pickc = next(iter(state_space.root.children[pickctable]))
    s_pickd = next(iter(state_space.root.children[pickdtable]))
    assert s_picka.alive == Alive.ALIVE
    assert s_pickb.alive == Alive.DEAD
    assert s_pickc.alive == Alive.DEAD
    assert s_pickd.alive == Alive.DEAD
    assert len(s_pickb.children) == 0
    assert len(s_pickc.children) == 0
    assert len(s_pickd.children) == 0
    assert len(s_picka.children) == 5
    putatable = get_action(ground_actions, "put", (a, table))
    putaa = get_action(ground_actions, "put", (a, a))
    putab = get_action(ground_actions, "put", (a, b))
    putac = get_action(ground_actions, "put", (a, c))
    putad = get_action(ground_actions, "put", (a, d))
    assert set(s_picka.children.keys()) == {putatable, putaa, putab, putac, putad}
    s_putatable = next(iter(s_picka.children[putatable]))
    s_putaa = next(iter(s_picka.children[putaa]))
    s_putab = next(iter(s_picka.children[putab]))
    s_putac = next(iter(s_picka.children[putac]))
    s_putad = next(iter(s_picka.children[putad]))
    assert s_putatable.alive == Alive.ALIVE  # initial state
    assert s_putaa.alive == Alive.DEAD
    assert s_putab.alive == Alive.ALIVE
    assert s_putac.alive == Alive.DEAD
    assert s_putad.alive == Alive.DEAD


def off_plan_nodes(state_space, problem):
    """The plan-restricted fringe: successors of on-plan states that no plan prescribes."""
    return [
        node
        for node in state_space.nodes.values()
        if not node.children and not check_formula(node.state, problem.goal)
    ]


def test_plan_input_frontier_marks_pruned(typed_blocks_medsize):
    domain, problem = typed_blocks_medsize
    plan = PlanParser()("(pick a table) (put a b)")
    without = generate_state_space(domain, problem, plans=[plan])
    with_frontier = generate_state_space(domain, problem, plans=[plan], frontier=True)
    # The flag only relabels the fringe; it must not expand anything extra.
    assert len(with_frontier.nodes) == len(without.nodes)
    fringe = off_plan_nodes(with_frontier, problem)
    assert fringe
    assert all(node.alive == Alive.PRUNED for node in fringe)
    assert all(node.alive == Alive.DEAD for node in off_plan_nodes(without, problem))


def test_off_plan_goal_successor_is_alive(typed_blocks_medsize):
    domain, problem = typed_blocks_medsize
    plan = PlanParser()("(pick a table) (put a b)")
    for frontier in (False, True):
        state_space = generate_state_space(domain, problem, plans=[plan], frontier=frontier)
        goal_nodes = [n for n in state_space.nodes.values() if check_formula(n.state, problem.goal)]
        # There are off-plan successors that already satisfy the goal. They are never queued,
        # so the goal check in the expansion loop never runs on them; classifying them at
        # creation time keeps them out of the frontier (and out of the planner).
        assert len(goal_nodes) > 1, f"fixture no longer exercises the case (frontier={frontier})"
        assert all(n.alive == Alive.ALIVE and n.goal for n in goal_nodes)
        unexpanded = [n for n in goal_nodes if not n.children]
        assert unexpanded, "off-plan goal successors should not be expanded"


def test_dead_states_are_not_expanded(typed_blocks_medsize):
    domain, problem = typed_blocks_medsize
    plan = PlanParser()("(pick a table) (put a b)")
    ground_actions = ground(domain, problem)
    b = Constant("b", "block")
    table = Constant("table", "obj")
    pickbtable = get_action(ground_actions, "pick", (b, table))
    baseline = generate_state_space(domain, problem, plans=[plan], frontier=True)
    off_plan = next(iter(baseline.root.children[pickbtable]))
    assert off_plan.alive == Alive.PRUNED

    refuted = generate_state_space(domain, problem, plans=[plan], frontier=True, dead_states={off_plan.state})
    node = next(iter(refuted.root.children[pickbtable]))
    # A refuted state emits no pruned/2 fact, so the solver cannot route through it.
    assert node.alive == Alive.DEAD
    assert not node.children


def test_action_path_from_root(typed_blocks_medsize):
    domain, problem = typed_blocks_medsize
    plan = PlanParser()("(pick a table) (put a b)")
    state_space = generate_state_space(domain, problem, plans=[plan], frontier=True)
    assert state_space.action_path_from_root(state_space.root) == []
    for node in state_space.nodes.values():
        path = state_space.action_path_from_root(node)
        assert path is not None, f"node {node.id} is unreachable from the root"
        # Replaying the path from the initial state must land on the node's state.
        state = problem.init
        for action in path:
            assert check_formula(state, action.precondition)
            state = next(iter(apply_action_effects(state, action)))
        assert state == node.state


def test_every_state_on_a_plan_is_expanded(gripper):
    """A state that lies on an example plan must be expanded.

    States are expanded in a single LIFO pass, so a plan can reach a node that was already
    expanded. Unless the suffix propagation is iterated, the actions that plan prescribes
    there are never matched and its successors are wrongly left off-plan. These plans, which
    reconverge on shared states, trigger that ordering; the last one is the victim.
    """
    domain, problem = gripper
    plans = [
        PlanParser()(p)
        for p in [
            "(pick ball1 rooma left) (move rooma roomb) (drop ball1 roomb left)"
            " (move roomb rooma) (pick ball2 rooma left) (move rooma roomb) (drop ball2 roomb left)",
            "(pick ball1 rooma left) (move rooma roomb) (drop ball1 roomb left) (move roomb rooma)"
            " (pick ball2 rooma left) (move rooma roomb) (drop ball2 roomb left) (move roomb rooma)",
            "(pick ball2 rooma left) (move rooma roomb) (drop ball2 roomb left)"
            " (move roomb rooma) (pick ball1 rooma left) (move rooma roomb) (drop ball1 roomb left)",
            "(pick ball2 rooma left) (move rooma roomb) (drop ball2 roomb left) (move roomb rooma)"
            " (pick ball1 rooma left) (move rooma roomb) (drop ball1 roomb left) (move roomb rooma)",
            "(pick ball1 rooma left) (move rooma roomb) (drop ball1 roomb left)"
            " (move roomb rooma) (pick ball2 rooma right) (move rooma roomb) (drop ball2 roomb right)",
            "(move rooma roomb) (move roomb rooma) (pick ball1 rooma left) (move rooma roomb)"
            " (drop ball1 roomb left) (move roomb rooma) (pick ball2 rooma left)"
            " (move rooma roomb) (drop ball2 roomb left)",
            "(pick ball1 rooma right) (move rooma roomb) (drop ball1 roomb right)"
            " (move roomb rooma) (pick ball2 rooma left) (move rooma roomb) (drop ball2 roomb left)",
        ]
    ]
    state_space = generate_state_space(domain, problem, plans=plans, frontier=True)
    for i, plan in enumerate(plans):
        state = problem.init
        for step, action in enumerate(plan.instantiate(domain)):
            state = next(iter(apply_action_effects(state, action)))
            node = state_space.nodes.get(state)
            assert node is not None, f"plan {i} step {step}: state missing from the graph"
            assert node.alive == Alive.ALIVE, (
                f"plan {i} step {step} after {action.name}: state lies on an example plan" f" but is {node.alive.name}"
            )
