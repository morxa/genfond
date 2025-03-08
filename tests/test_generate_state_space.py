from helpers import get_action
from pddl.logic import Predicate, constants

from genfond.ground import ground
from genfond.state_space_generator import apply_action_effects, generate_state_space


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
