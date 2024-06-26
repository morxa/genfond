from genfond.feature_generator import FeaturePool
from genfond.state_space_generator import apply_action_effects, check_formula
from genfond.ground import ground
from pddl.logic import constants, Predicate
from helpers import get_action


def test_generate_features_simple_blocks(simple_blocks):
    domain, problem = simple_blocks
    feature_pool = FeaturePool(domain, [problem], all_generators=True)
    a, b = constants('a b')
    assert 'b_empty(c_primitive(holding,0))' in feature_pool.features
    assert 'b_empty(r_primitive(on,0,1))' in feature_pool.features
    on_ab = frozenset([Predicate('on', a, b)])
    print('\n'.join([f'{k} = {feature_pool.evaluate_feature(k, on_ab)}' for k in feature_pool.features.keys()]))
    assert feature_pool.evaluate_feature('b_empty(c_primitive(holding,0))', on_ab) is True
    assert feature_pool.evaluate_feature('b_empty(r_primitive(on,0,1))', on_ab) is False
    assert feature_pool.evaluate_feature('n_count(c_primitive(holding,0))', on_ab) == 0
    assert feature_pool.evaluate_feature('n_count(r_primitive(on,0,1))', on_ab) == 1
    assert feature_pool.evaluate_feature('n_count(r_primitive(on_G,0,1))', on_ab) == 1
    assert feature_pool.evaluate_feature('n_count(r_transitive_reflexive_closure(r_primitive(on,0,1)))', on_ab) == 4
    holding_a = frozenset([Predicate('holding', a)])
    assert feature_pool.evaluate_feature('b_empty(c_primitive(holding,0))', holding_a) is False
    assert feature_pool.evaluate_feature('b_empty(r_primitive(on,0,1))', holding_a) is True
    assert feature_pool.evaluate_feature('n_count(c_primitive(holding,0))', holding_a) == 1
    assert feature_pool.evaluate_feature('n_count(r_primitive(on,0,1))', holding_a) == 0
    assert feature_pool.evaluate_feature('n_count(r_primitive(on_G,0,1))', holding_a) == 1
    assert feature_pool.evaluate_feature('n_count(r_transitive_reflexive_closure(r_primitive(on,0,1)))',
                                         holding_a) == 3


def test_generate_features_fond_blocks(fond_blocks):
    domain, problem = fond_blocks
    ground_actions = ground(domain, problem)
    feature_pool = FeaturePool(domain, [problem], all_generators=False)
    gstates = [
        node.state for node in feature_pool.state_graphs[problem.name].nodes.values()
        if check_formula(node.state, problem.goal)
    ]
    assert len(gstates) > 0
    a, b, c, table = constants('A B C Table')
    istate = feature_pool.state_graphs[problem.name].root.state
    print('\n'.join([f'{k} = {feature_pool.evaluate_feature(k, istate)}' for k in feature_pool.features.keys()]))
    assert feature_pool.evaluate_feature('n_count(r_primitive(on,0,1))', istate) == 3
    assert feature_pool.evaluate_feature('n_count(c_primitive(clear,0))', istate) == 3
    assert feature_pool.evaluate_feature('n_count(c_some(r_primitive(on,0,1),c_one_of(Table)))', istate) == 2
    assert feature_pool.evaluate_feature('n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))', istate) == 0
    cta_puton = get_action(ground_actions, 'puton', (c, table, a))
    cta_puton_state = next(iter(apply_action_effects(istate, cta_puton)))
    assert feature_pool.evaluate_feature('n_count(c_primitive(clear,0))', cta_puton_state) == 4
    assert feature_pool.evaluate_feature('n_count(c_some(r_primitive(on,0,1),c_one_of(Table)))', cta_puton_state) == 3
    assert feature_pool.evaluate_feature('n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))',
                                         cta_puton_state) == 0
    assert feature_pool.evaluate_feature('n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))', gstates[0]) == 1


def test_features_to_clingo(simple_blocks):
    domain, problem = simple_blocks
    feature_pool = FeaturePool(domain, [problem], 2)
    clingo_program = feature_pool.to_clingo()
    assert 'feature("b_empty(c_primitive(holding,0))").' in clingo_program
    assert 'feature_complexity("b_empty(c_primitive(holding,0))", 2).' in clingo_program
    assert 'feature("b_empty(r_primitive(on,0,1))").' in clingo_program
    assert 'feature_complexity("b_empty(r_primitive(on,0,1))", 2).' in clingo_program
    assert 'feature("n_count(r_primitive(on,0,1))").' in clingo_program
    assert 'feature_complexity("n_count(r_primitive(on,0,1))", 2).' in clingo_program
    assert 'feature("n_count(r_primitive(on_G,0,1))").' in clingo_program
    assert 'feature_complexity("n_count(r_primitive(on_G,0,1))", 2).' in clingo_program
    assert 'state(0, 0).' in clingo_program
    assert 'eval(0, 0, "b_empty(c_primitive(holding,0))", 1).' in clingo_program
    assert 'eval(0, 0, "b_empty(r_primitive(on,0,1))", 0).' in clingo_program
    assert 'eval(0, 0, "n_count(r_primitive(on,0,1))", 1).' in clingo_program
    assert 'eval(0, 0, "n_count(r_primitive(on_G,0,1))", 1).' in clingo_program
    assert 'state(0, 1).' in clingo_program
    assert 'eval(0, 1, "b_empty(c_primitive(holding,0))", 0).' in clingo_program
    assert 'eval(0, 1, "b_empty(r_primitive(on,0,1))", 1).' in clingo_program
    assert 'eval(0, 1, "n_count(r_primitive(on,0,1))", 0).' in clingo_program
    assert 'eval(0, 1, "n_count(r_primitive(on_G,0,1))", 1).' in clingo_program
    assert 'state(0, 2).' in clingo_program
    assert 'eval(0, 1, "b_empty(c_primitive(holding,0))", 0).' in clingo_program
    assert 'eval(0, 1, "b_empty(r_primitive(on,0,1))", 1).' in clingo_program
    assert 'eval(0, 1, "n_count(r_primitive(on,0,1))", 0).' in clingo_program
    assert 'eval(0, 2, "n_count(r_primitive(on_G,0,1))", 1).' in clingo_program
    assert 'trans(0, 0, "pick(a,b)", 1).' in clingo_program
    assert 'trans(0, 1, "put(a,b)", 0).' in clingo_program


def test_features_clingo_upper_case(fond_blocks):
    domain, problem = fond_blocks
    feature_pool = FeaturePool(domain, [problem], 2)
    clingo_program = feature_pool.to_clingo()
    # Omit the target state as it might vary.
    assert 'trans(0, 0, "puton(B,C,Table)"' in clingo_program
