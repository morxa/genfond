from genfond.feature_generator import FeaturePool
from genfond.state_space_generator import apply_action_effects, check_formula
from genfond.ground import ground
from genfond.config_handler import ConfigHandler
from pddl.logic import constants, Predicate
from helpers import get_action
import re
import pytest


def test_generate_features_simple_blocks(simple_blocks):
    domain, problem = simple_blocks
    config = ConfigHandler(type='datalog')
    config['include_numerical_features'] = True
    config['feature_generator']['generate_count_numerical'] = True
    feature_pool = FeaturePool(domain, [problem], config=config, all_generators=True)
    a, b, c = constants('a b c')
    assert 'b_empty(c_primitive(holding,0))' in feature_pool.features
    assert 'b_empty(r_primitive(on,0,1))' in feature_pool.features
    on_ab = frozenset([Predicate('on', a, b)])
    #print('\n'.join([f'{k} = {feature_pool.evaluate_concept(k, on_ab)}' for k in feature_pool.concepts.keys()]))
    #print('\n'.join([f'{k} = {feature_pool.evaluate_feature(k, on_ab)}' for k in feature_pool.features.keys()]))
    assert feature_pool.evaluate_feature('b_empty(c_primitive(holding,0))', problem, on_ab) is True
    assert feature_pool.evaluate_feature('b_empty(r_primitive(on,0,1))', problem, on_ab) is False
    assert feature_pool.evaluate_feature('n_count(c_primitive(holding,0))', problem, on_ab) == 0
    assert feature_pool.evaluate_feature('n_count(r_primitive(on,0,1))', problem, on_ab) == 1
    assert feature_pool.evaluate_feature('n_count(r_primitive(on_G,0,1))', problem, on_ab) == 1
    assert feature_pool.evaluate_feature('n_count(r_transitive_reflexive_closure(r_primitive(on,0,1)))', problem,
                                         on_ab) == 4
    holding_a = frozenset([Predicate('holding', a)])
    assert feature_pool.evaluate_feature('b_empty(c_primitive(holding,0))', problem, holding_a) is False
    assert feature_pool.evaluate_feature('b_empty(r_primitive(on,0,1))', problem, holding_a) is True
    assert feature_pool.evaluate_feature('n_count(c_primitive(holding,0))', problem, holding_a) == 1
    assert feature_pool.evaluate_feature('n_count(r_primitive(on,0,1))', problem, holding_a) == 0
    assert feature_pool.evaluate_feature('n_count(r_primitive(on_G,0,1))', problem, holding_a) == 1
    assert feature_pool.evaluate_feature('n_count(r_transitive_reflexive_closure(r_primitive(on,0,1)))', problem,
                                         holding_a) == 3
    assert feature_pool.evaluate_concept('c_top', problem, on_ab) == {a.name, b.name, c.name}
    assert feature_pool.evaluate_concept('c_primitive(holding,0)', problem, on_ab) == set()
    assert feature_pool.evaluate_concept('c_primitive(holding,0)', problem, holding_a) == {a.name}
    assert feature_pool.evaluate_role('r_primitive(on,0,1)', problem, on_ab) == {(a.name, b.name)}
    assert feature_pool.evaluate_role('r_primitive(on,0,1)', problem, holding_a) == set()
    assert feature_pool.evaluate_role('r_primitive(on_G,0,1)', problem, on_ab) == {(a.name, c.name)}


@pytest.mark.skip(reason='r_and is not generated')
def test_generate_features_fond_blocks(fond_blocks):
    domain, problem = fond_blocks
    ground_actions = ground(domain, problem)
    config = ConfigHandler(type='datalog')
    config['include_numerical_features'] = True
    feature_pool = FeaturePool(domain, [problem], config=config, all_generators=False)
    gstates = [
        node.state for node in feature_pool.state_graphs[problem.name].nodes.values()
        if check_formula(node.state, problem.goal)
    ]
    assert len(gstates) > 0
    a, b, c, table = constants('A B C Table')
    istate = feature_pool.state_graphs[problem.name].root.state
    assert feature_pool.evaluate_feature('n_count(r_primitive(on,0,1))', problem, istate) == 3
    assert feature_pool.evaluate_feature('n_count(c_primitive(clear,0))', problem, istate) == 3
    assert feature_pool.evaluate_feature('n_count(c_some(r_primitive(on,0,1),c_one_of(Table)))', problem, istate) == 2
    assert feature_pool.evaluate_feature('n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))', problem,
                                         istate) == 0
    cta_puton = get_action(ground_actions, 'puton', (c, table, a))
    cta_puton_state = next(iter(apply_action_effects(istate, cta_puton)))
    assert feature_pool.evaluate_feature('n_count(c_primitive(clear,0))', problem, cta_puton_state) == 4
    assert feature_pool.evaluate_feature('n_count(c_some(r_primitive(on,0,1),c_one_of(Table)))', problem,
                                         cta_puton_state) == 3
    assert feature_pool.evaluate_feature('n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))', problem,
                                         cta_puton_state) == 0
    assert feature_pool.evaluate_feature('n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))', problem,
                                         gstates[0]) == 1
    assert feature_pool.evaluate_concept('c_primitive(clear,0)', problem, istate) == {table.name, b.name, c.name}
    assert feature_pool.evaluate_concept('c_some(r_primitive(on,0,1),c_one_of(Table))', problem,
                                         istate) == {a.name, b.name}
    assert feature_pool.evaluate_role('r_and(r_primitive(on,0,1),r_primitive(on_G,0,1))', problem,
                                      gstates[0]) == {(b.name, a.name)}


def test_features_to_clingo(simple_blocks):
    domain, problem = simple_blocks
    config = ConfigHandler()
    config['max_complexity'] = 2
    config['prune_features'] = False
    feature_pool = FeaturePool(domain, [problem], config=config)
    clingo_program = feature_pool.to_clingo()
    print(f'full program:\n{clingo_program}')
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
    assert 'c_eval' not in clingo_program


def test_concepts_to_clingo(simple_blocks):
    domain, problem = simple_blocks
    config = ConfigHandler()
    config['include_boolean_features'] = False
    config['include_numerical_features'] = False
    config['include_concepts'] = True
    config['max_complexity'] = 2
    config['prune_concepts'] = False
    config['prune_roles'] = False
    feature_pool = FeaturePool(domain, [problem], config=config)
    clingo_program = feature_pool.to_clingo()
    print(f'full program:\n{clingo_program}')
    assert 'concept("c_top").' in clingo_program
    assert 'concept_complexity("c_top", 1).' in clingo_program
    assert 'concept("c_primitive(holding,0)").' in clingo_program
    assert 'concept_complexity("c_primitive(holding,0)", 1).' in clingo_program
    assert 'concept("c_not(c_primitive(holding,0))").' in clingo_program
    assert 'concept_complexity("c_not(c_primitive(holding,0))", 2).' in clingo_program
    assert 'state(0, 0).' in clingo_program
    assert 'state(0, 1).' in clingo_program
    assert 'state(0, 2).' in clingo_program
    assert 'state(0, 3).' in clingo_program
    assert 'trans(0, 0, "pick(a,b)", 1).' in clingo_program
    assert 'trans(0, 1, "put(a,b)", 0).' in clingo_program
    assert 'trans(0, 1, "put(a,a)", 2)' in clingo_program
    assert 'trans(0, 1, "put(a,c)", 3)' in clingo_program
    assert 'c_eval(0, 0, "c_top", "a").' in clingo_program
    assert 'c_eval(0, 0, "c_top", "b").' in clingo_program
    assert 'c_eval(0, 0, "c_top", "c").' not in clingo_program
    assert 'c_eval(0, 1, "c_primitive(holding,0)", "a").' in clingo_program
    assert 'c_eval(0, 1, "c_top", "a").' in clingo_program
    assert 'c_eval(0, 1, "c_top", "b").' in clingo_program
    assert 'c_eval(0, 1, "c_top", "c").' in clingo_program
    assert 'c_eval(0, 2, "c_top", "a").' in clingo_program
    assert 'c_eval(0, 2, "c_top", "b").' not in clingo_program
    assert 'c_eval(0, 2, "c_top", "c").' not in clingo_program
    assert 'c_eval(0, 3, "c_top", "a").' in clingo_program
    assert 'c_eval(0, 3, "c_top", "b").' not in clingo_program
    assert 'c_eval(0, 3, "c_top", "c").' in clingo_program
    assert not re.search(r'^eval.*', clingo_program, re.MULTILINE)


def test_features_clingo_upper_case(fond_blocks):
    domain, problem = fond_blocks
    config = ConfigHandler()
    config['max_complexity'] = 2
    feature_pool = FeaturePool(domain, [problem], config=config)
    clingo_program = feature_pool.to_clingo()
    # Omit the target state as it might vary.
    assert 'trans(0, 0, "puton(B,C,Table)"' in clingo_program


def test_features_with_augmented_states(blocks_clear):
    domain, problem = blocks_clear
    ground_actions = ground(domain, problem)
    config = ConfigHandler()
    config['include_actions'] = True
    config['max_complexity'] = 4
    feature_pool = FeaturePool(domain, [problem], config=config)
    b0, b1 = constants('b0 b1')
    unstack10 = get_action(ground_actions, 'unstack', (b1, b0))
    unstack01 = get_action(ground_actions, 'unstack', (b0, b1))
    for feature in feature_pool.features.keys():
        print(f'{feature} = {feature_pool.evaluate_feature(feature, problem, problem.init, unstack10)}')
    assert feature_pool.evaluate_feature('b_empty(c_and(c_primitive(clear_G,0),c_primitive(aparam1,0)))', problem,
                                         problem.init, unstack10) == False
    assert feature_pool.evaluate_feature('b_empty(c_and(c_primitive(clear_G,0),c_primitive(aparam1,0)))', problem,
                                         problem.init, unstack01) == True


def test_augmented_states_to_clingo(blocks_clear):
    domain, problem = blocks_clear
    ground_actions = ground(domain, problem)
    config = ConfigHandler()
    config['include_actions'] = True
    config['max_complexity'] = 4
    feature_pool = FeaturePool(domain, [problem], config=config)
    clingo_program = feature_pool.to_clingo()
    print(f'full program:\n{clingo_program}')
    assert 'aug_state(0, 0, "unstack(b1,b0)", 0).' in clingo_program
    assert 'eval(0, "b_empty(c_and(c_primitive(clear_G,0),c_primitive(aparam1,0)))", 0).' in clingo_program
    assert 'trans(0, 0, "unstack(b1,b0)", 1).' in clingo_program
