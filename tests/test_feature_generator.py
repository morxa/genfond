from genfond.feature_generator import FeaturePool
from genfond.state_space_generator import generate_state_space, \
        apply_action_effects, check_formula
from genfond.ground import ground
from pddl.logic import constants, Predicate
from helpers import get_action


def test_generate_features_simple_blocks(simple_blocks):
    domain, problem = simple_blocks
    state_graph = generate_state_space(domain, problem)
    states = [node.state for node in state_graph.nodes.values()]
    feature_pool = FeaturePool(domain, [problem])
    a, b = constants('a b')
    assert 'b_empty(c_primitive(holding,0))' in feature_pool.features
    assert 'b_empty(r_primitive(on,0,1))' in feature_pool.features
    on_ab = frozenset([Predicate('on', a, b)])
    assert feature_pool.evaluate_feature('b_empty(c_primitive(holding,0))',
                                         on_ab) is True
    assert feature_pool.evaluate_feature('b_empty(r_primitive(on,0,1))',
                                         on_ab) is False
    assert feature_pool.evaluate_feature('n_count(c_primitive(holding,0))',
                                         on_ab) == 0
    assert feature_pool.evaluate_feature('n_count(r_primitive(on,0,1))',
                                         on_ab) == 1
    assert feature_pool.evaluate_feature('n_count(r_primitive(on_G,0,1))',
                                         on_ab) == 1
    assert feature_pool.evaluate_feature(
        'n_count(c_all(r_primitive(on,0,1),'
        'c_some(r_primitive(on_G,0,1),c_top)))', on_ab) == 2
    holding_a = frozenset([Predicate('holding', a)])
    assert feature_pool.evaluate_feature('b_empty(c_primitive(holding,0))',
                                         holding_a) is False
    assert feature_pool.evaluate_feature('b_empty(r_primitive(on,0,1))',
                                         holding_a) is True
    assert feature_pool.evaluate_feature('n_count(c_primitive(holding,0))',
                                         holding_a) == 1
    assert feature_pool.evaluate_feature('n_count(r_primitive(on,0,1))',
                                         holding_a) == 0
    assert feature_pool.evaluate_feature('n_count(r_primitive(on_G,0,1))',
                                         holding_a) == 1
    assert feature_pool.evaluate_feature(
        'n_count(c_all(r_primitive(on,0,1),'
        'c_some(r_primitive(on_G,0,1),c_top)))', holding_a) == 2


def test_generate_features_fond_blocks(fond_blocks):
    domain, problem = fond_blocks
    ground_actions = ground(domain, problem)
    state_graph = generate_state_space(domain, problem)
    states = [node.state for node in state_graph.nodes.values()]
    gstates = [
        node.state for node in state_graph.nodes.values()
        if check_formula(node.state, problem.goal)
    ]
    assert len(gstates) > 0
    feature_pool = FeaturePool(domain, [problem])
    a, b, c, table = constants('A B C Table')
    istate = state_graph.root.state
    assert feature_pool.evaluate_feature('n_count(r_primitive(on,0,1))',
                                         istate) == 3
    assert feature_pool.evaluate_feature('n_count(c_primitive(clear,0))',
                                         istate) == 3
    assert feature_pool.evaluate_feature(
        'n_count(c_some(r_primitive(on,0,1),c_one_of(Table)))', istate) == 2
    assert feature_pool.evaluate_feature(
        'n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))',
        istate) == 0
    cta_puton = get_action(ground_actions, 'puton', (c, table, a))
    cta_puton_state = next(iter(apply_action_effects(istate, cta_puton)))
    assert feature_pool.evaluate_feature('n_count(c_primitive(clear,0))',
                                         cta_puton_state) == 4
    assert feature_pool.evaluate_feature(
        'n_count(c_some(r_primitive(on,0,1),c_one_of(Table)))',
        cta_puton_state) == 3
    assert feature_pool.evaluate_feature(
        'n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))',
        cta_puton_state) == 0
    assert feature_pool.evaluate_feature(
        'n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))',
        gstates[0]) == 1
