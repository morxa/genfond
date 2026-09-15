import re

import pytest
from pddl.logic import Predicate, constants

from genfond.config_handler import ConfigHandler
from genfond.feature_generator import FeaturePool
from genfond.ground import ground
from genfond.state_space_generator import apply_action_effects, check_formula

from .helpers import get_action


def test_generate_features_simple_blocks(simple_blocks):
    domain, problem = simple_blocks
    config = ConfigHandler(type="datalog")
    config["include_numerical_features"] = True
    config["feature_generator"]["generate_count_numerical"] = True
    feature_pool = FeaturePool(domain, [problem], config=config, all_generators=True)
    a, b, c = constants("a b c")
    assert "b_empty(c_primitive(holding,0))" in feature_pool.features
    assert "b_empty(r_primitive(on,0,1))" in feature_pool.features
    on_ab = frozenset([Predicate("on", a, b)])
    # print('\n'.join([f'{k} = {feature_pool.evaluate_concept(k, on_ab)}' for k in feature_pool.concepts.keys()]))
    # print('\n'.join([f'{k} = {feature_pool.evaluate_feature(k, on_ab)}' for k in feature_pool.features.keys()]))
    assert feature_pool.evaluate_feature("b_empty(c_primitive(holding,0))", problem, on_ab) is True
    assert feature_pool.evaluate_feature("b_empty(r_primitive(on,0,1))", problem, on_ab) is False
    assert feature_pool.evaluate_feature("n_count(c_primitive(holding,0))", problem, on_ab) == 0
    assert feature_pool.evaluate_feature("n_count(r_primitive(on,0,1))", problem, on_ab) == 1
    assert feature_pool.evaluate_feature("n_count(r_primitive(on_G,0,1))", problem, on_ab) == 1
    assert (
        feature_pool.evaluate_feature("n_count(r_transitive_reflexive_closure(r_primitive(on,0,1)))", problem, on_ab)
        == 4
    )
    holding_a = frozenset([Predicate("holding", a)])
    assert feature_pool.evaluate_feature("b_empty(c_primitive(holding,0))", problem, holding_a) is False
    assert feature_pool.evaluate_feature("b_empty(r_primitive(on,0,1))", problem, holding_a) is True
    assert feature_pool.evaluate_feature("n_count(c_primitive(holding,0))", problem, holding_a) == 1
    assert feature_pool.evaluate_feature("n_count(r_primitive(on,0,1))", problem, holding_a) == 0
    assert feature_pool.evaluate_feature("n_count(r_primitive(on_G,0,1))", problem, holding_a) == 1
    assert (
        feature_pool.evaluate_feature(
            "n_count(r_transitive_reflexive_closure(r_primitive(on,0,1)))", problem, holding_a
        )
        == 3
    )
    assert feature_pool.evaluate_concept("c_top", problem, on_ab) == {a.name, b.name, c.name}
    assert feature_pool.evaluate_concept("c_primitive(holding,0)", problem, on_ab) == set()
    assert feature_pool.evaluate_concept("c_primitive(holding,0)", problem, holding_a) == {a.name}
    assert feature_pool.evaluate_role("r_primitive(on,0,1)", problem, on_ab) == {(a.name, b.name)}
    assert feature_pool.evaluate_role("r_primitive(on,0,1)", problem, holding_a) == set()
    assert feature_pool.evaluate_role("r_primitive(on_G,0,1)", problem, on_ab) == {(a.name, c.name)}


@pytest.mark.skip(reason="r_and is not generated")
def test_generate_features_fond_blocks(fond_blocks):
    domain, problem = fond_blocks
    ground_actions = ground(domain, problem)
    config = ConfigHandler(type="datalog")
    config["include_numerical_features"] = True
    feature_pool = FeaturePool(domain, [problem], config=config, all_generators=False)
    gstates = [
        node.state
        for node in feature_pool.state_graphs[problem.name].nodes.values()
        if check_formula(node.state, problem.goal)
    ]
    assert len(gstates) > 0
    a, b, c, table = constants("A B C Table")
    istate = feature_pool.state_graphs[problem.name].root.state
    assert feature_pool.evaluate_feature("n_count(r_primitive(on,0,1))", problem, istate) == 3
    assert feature_pool.evaluate_feature("n_count(c_primitive(clear,0))", problem, istate) == 3
    assert feature_pool.evaluate_feature("n_count(c_some(r_primitive(on,0,1),c_one_of(Table)))", problem, istate) == 2
    assert (
        feature_pool.evaluate_feature("n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))", problem, istate)
        == 0
    )
    cta_puton = get_action(ground_actions, "puton", (c, table, a))
    cta_puton_state = next(iter(apply_action_effects(istate, cta_puton)))
    assert feature_pool.evaluate_feature("n_count(c_primitive(clear,0))", problem, cta_puton_state) == 4
    assert (
        feature_pool.evaluate_feature("n_count(c_some(r_primitive(on,0,1),c_one_of(Table)))", problem, cta_puton_state)
        == 3
    )
    assert (
        feature_pool.evaluate_feature(
            "n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))", problem, cta_puton_state
        )
        == 0
    )
    assert (
        feature_pool.evaluate_feature("n_count(r_and(r_primitive(on,0,1),r_primitive(on_G,0,1)))", problem, gstates[0])
        == 1
    )
    assert feature_pool.evaluate_concept("c_primitive(clear,0)", problem, istate) == {table.name, b.name, c.name}
    assert feature_pool.evaluate_concept("c_some(r_primitive(on,0,1),c_one_of(Table))", problem, istate) == {
        a.name,
        b.name,
    }
    assert feature_pool.evaluate_role("r_and(r_primitive(on,0,1),r_primitive(on_G,0,1))", problem, gstates[0]) == {
        (b.name, a.name)
    }


def test_features_to_clingo(simple_blocks):
    domain, problem = simple_blocks
    config = ConfigHandler()
    config["max_complexity"] = 2
    config["prune_features"] = False
    config["prune_redundant_features"] = False
    config["prune_redundant_concepts"] = False
    config["prune_redundant_roles"] = False
    feature_pool = FeaturePool(domain, [problem], config=config)
    clingo_program = feature_pool.to_clingo()
    print(f"full program:\n{clingo_program}")
    assert 'feature("b_empty(c_primitive(holding,0))").' in clingo_program
    assert 'feature_complexity("b_empty(c_primitive(holding,0))", 2).' in clingo_program
    assert 'feature("b_empty(r_primitive(on,0,1))").' in clingo_program
    assert 'feature_complexity("b_empty(r_primitive(on,0,1))", 2).' in clingo_program
    assert 'feature("n_count(r_primitive(on,0,1))").' in clingo_program
    assert 'feature_complexity("n_count(r_primitive(on,0,1))", 2).' in clingo_program
    assert 'feature("n_count(r_primitive(on_G,0,1))").' in clingo_program
    assert 'feature_complexity("n_count(r_primitive(on_G,0,1))", 2).' in clingo_program
    assert "state(0, 0)." in clingo_program
    assert 'eval(0, 0, "b_empty(c_primitive(holding,0))", 1).' in clingo_program
    assert 'eval(0, 0, "b_empty(r_primitive(on,0,1))", 0).' in clingo_program
    assert 'eval(0, 0, "n_count(r_primitive(on,0,1))", 1).' in clingo_program
    assert 'eval(0, 0, "n_count(r_primitive(on_G,0,1))", 1).' in clingo_program
    assert "state(0, 1)." in clingo_program
    assert 'eval(0, 1, "b_empty(c_primitive(holding,0))", 0).' in clingo_program
    assert 'eval(0, 1, "b_empty(r_primitive(on,0,1))", 1).' in clingo_program
    assert 'eval(0, 1, "n_count(r_primitive(on,0,1))", 0).' in clingo_program
    assert 'eval(0, 1, "n_count(r_primitive(on_G,0,1))", 1).' in clingo_program
    assert "state(0, 2)." in clingo_program
    assert 'eval(0, 1, "b_empty(c_primitive(holding,0))", 0).' in clingo_program
    assert 'eval(0, 1, "b_empty(r_primitive(on,0,1))", 1).' in clingo_program
    assert 'eval(0, 1, "n_count(r_primitive(on,0,1))", 0).' in clingo_program
    assert 'eval(0, 2, "n_count(r_primitive(on_G,0,1))", 1).' in clingo_program
    assert 'trans(0, 0, "pick(a,b)", 1).' in clingo_program
    assert 'trans(0, 1, "put(a,b)", 0).' in clingo_program
    assert "c_eval" not in clingo_program


def test_concepts_to_clingo(simple_blocks):
    domain, problem = simple_blocks
    config = ConfigHandler()
    config["include_boolean_features"] = False
    config["include_numerical_features"] = False
    config["include_concepts"] = True
    config["max_complexity"] = 2
    config["prune_concepts"] = False
    config["prune_roles"] = False
    config["prune_static_concepts"] = False
    config["prune_static_roles"] = False
    feature_pool = FeaturePool(domain, [problem], config=config)
    clingo_program = feature_pool.to_clingo()
    print(f"full program:\n{clingo_program}")
    assert 'concept("c_top").' in clingo_program
    assert 'concept_complexity("c_top", 1).' in clingo_program
    assert 'concept("c_primitive(holding,0)").' in clingo_program
    assert 'concept_complexity("c_primitive(holding,0)", 1).' in clingo_program
    assert 'concept("c_not(c_primitive(holding,0))").' in clingo_program
    assert 'concept_complexity("c_not(c_primitive(holding,0))", 2).' in clingo_program
    assert "state(0, 0)." in clingo_program
    assert "state(0, 1)." in clingo_program
    assert "state(0, 2)." in clingo_program
    assert "state(0, 3)." in clingo_program
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
    assert not re.search(r"^eval.*", clingo_program, re.MULTILINE)


def test_features_clingo_upper_case(fond_blocks):
    domain, problem = fond_blocks
    config = ConfigHandler()
    config["max_complexity"] = 2
    feature_pool = FeaturePool(domain, [problem], config=config)
    clingo_program = feature_pool.to_clingo()
    # Omit the target state as it might vary.
    assert 'trans(0, 0, "puton(B,C,Table)"' in clingo_program


def test_features_with_augmented_states(blocks_clear):
    domain, problem = blocks_clear
    ground_actions = ground(domain, problem)
    config = ConfigHandler()
    config["include_actions"] = True
    config["max_complexity"] = 4
    feature_pool = FeaturePool(domain, [problem], config=config)
    b0, b1 = constants("b0 b1")
    unstack10 = get_action(ground_actions, "unstack", (b1, b0))
    unstack01 = get_action(ground_actions, "unstack", (b0, b1))
    for feature in feature_pool.features.keys():
        print(f"{feature} = {feature_pool.evaluate_feature(feature, problem, problem.init, unstack10)}")
    assert (
        feature_pool.evaluate_feature(
            "b_empty(c_and(c_primitive(clear_G,0),c_primitive(aparam1,0)))", problem, problem.init, unstack10
        )
        == False
    )
    assert (
        feature_pool.evaluate_feature(
            "b_empty(c_and(c_primitive(clear_G,0),c_primitive(aparam1,0)))", problem, problem.init, unstack01
        )
        == True
    )


def test_augmented_states_to_clingo(blocks_clear):
    domain, problem = blocks_clear
    ground_actions = ground(domain, problem)
    config = ConfigHandler()
    config["include_actions"] = True
    config["max_complexity"] = 4
    feature_pool = FeaturePool(domain, [problem], config=config)
    clingo_program = feature_pool.to_clingo()
    print(f"full program:\n{clingo_program}")
    assert 'aug_state(0, 0, "unstack(b1,b0)", 0).' in clingo_program
    assert 'eval(0, "b_empty(c_and(c_primitive(clear_G,0),c_primitive(aparam1,0)))", 0).' in clingo_program
    assert 'trans(0, 0, "unstack(b1,b0)", 1).' in clingo_program


def test_frontier_states_are_serialized_as_pruned(typed_blocks_medsize):
    from pddl.parser.plan import PlanParser

    from genfond.state_space_generator import Alive

    domain, problem = typed_blocks_medsize
    plan = PlanParser()("(pick a table) (put a b)")
    config = ConfigHandler(type="datalog")
    assert config["frontier_expansion"] is True
    feature_pool = FeaturePool(domain, [problem], config=config, plans={problem.name: [plan]})
    program = feature_pool.to_clingo()

    state_graph = feature_pool.state_graphs[problem.name]
    frontier_nodes = [n for n in state_graph.nodes.values() if n.alive == Alive.PRUNED]
    assert frontier_nodes
    for node in frontier_nodes:
        assert f"pruned(0, {node.id}).\n" in program
        # A frontier state is unexpanded: it carries no obligations for the policy.
        assert f"alive(0, {node.id}).\n" not in program
        assert f'eval(0, {node.id}, "' not in program
        # But the transition into it must be visible, or the solver cannot select it.
        assert f", {node.id}).\n" in program.replace(f"pruned(0, {node.id}).", "")

    # Refuting a frontier state removes its pruned/2 fact, which is what stops the solver
    # from routing through it.
    refuted = frontier_nodes[0].state
    blocked_pool = FeaturePool(
        domain,
        [problem],
        config=config,
        plans={problem.name: [plan]},
        dead_states={problem.name: {refuted}},
    )
    blocked_program = blocked_pool.to_clingo()
    refuted_id = blocked_pool.state_graphs[problem.name].nodes[refuted].id
    assert f"state(0, {refuted_id}).\n" in blocked_program
    assert f"pruned(0, {refuted_id}).\n" not in blocked_program


def test_lookup_node_resolves_asp_ids(typed_blocks_medsize):
    domain, problem = typed_blocks_medsize
    config = ConfigHandler(type="datalog")
    feature_pool = FeaturePool(domain, [problem], config=config)
    for node in feature_pool.state_graphs[problem.name].nodes.values():
        found_problem, found_node = feature_pool.lookup_node(0, node.id)
        assert found_problem.name == problem.name
        assert found_node is node


def test_concept_and_role_complexity_offsets_reach_dlplan(simple_blocks, monkeypatch):
    """concept_complexity_offset/role_complexity_offset must lower only the concept/role limits
    passed to dlplan's generate_features; boolean/numerical limits stay at the round's
    complexity. Positional order is (concept, role, boolean, count_numerical,
    distance_numerical) -- see dlplan/generator/__init__.pyi."""
    domain, problem = simple_blocks
    config = ConfigHandler(type="datalog")
    config["concept_complexity_offset"] = 2
    config["role_complexity_offset"] = 1
    max_complexity = 5

    captured_args = {}

    def fake_generate_features(factory, states, *limits, **kwargs):
        captured_args["limits"] = limits
        return [], [], [], []

    monkeypatch.setattr("genfond.feature_generator.dlplan_gen.generate_features", fake_generate_features)
    FeaturePool(domain, [problem], config=config, max_complexity=max_complexity)

    concept_limit, role_limit, boolean_limit, count_numerical_limit, distance_numerical_limit = captured_args[
        "limits"
    ][:5]
    assert concept_limit == max_complexity - 2
    assert role_limit == max_complexity - 1
    assert boolean_limit == max_complexity
    assert count_numerical_limit == max_complexity
    assert distance_numerical_limit == max_complexity


def test_concept_and_role_complexity_offsets_are_floored_at_one(simple_blocks, monkeypatch):
    domain, problem = simple_blocks
    config = ConfigHandler(type="datalog")
    config["concept_complexity_offset"] = 10
    config["role_complexity_offset"] = 10
    max_complexity = 3

    captured_args = {}

    def fake_generate_features(factory, states, *limits, **kwargs):
        captured_args["limits"] = limits
        return [], [], [], []

    monkeypatch.setattr("genfond.feature_generator.dlplan_gen.generate_features", fake_generate_features)
    FeaturePool(domain, [problem], config=config, max_complexity=max_complexity)

    concept_limit, role_limit = captured_args["limits"][:2]
    assert concept_limit == 1
    assert role_limit == 1


# The concept "has a goal support", i.e. is not ontable in the goal. On the state space of a
# single 3-block instance it has the same denotation as a shallower concept, so dlplan drops it;
# with the sample it survives. See docs/rich-sample-results.md.
SAMPLE_ONLY_CONCEPT = "c_some(r_primitive(on_G,0,1),c_top)"


def _denotations(concept, states):
    return [tuple(concept.evaluate(state).to_sorted_vector()) for state in states]


def _sample_pool(domain, train, all_problems, enabled, max_complexity=3, **kwargs):
    config = ConfigHandler(type="datalog-sig")
    config["feature_sample"]["enabled"] = enabled
    config["feature_sample"]["walks_per_problem"] = 3
    config["feature_sample"]["walk_length"] = 20
    config.update(kwargs)
    return FeaturePool(
        domain,
        train,
        config=config,
        max_complexity=max_complexity,
        all_generators=False,
        all_problems=all_problems,
    )


def test_feature_sample_keeps_concept_that_coincides_on_training_states(blocks3ops_small, blocks3ops):
    """A concept dlplan deduplicates away on the training states survives with the sample."""
    domain, small = blocks3ops_small
    _, large = blocks3ops
    without = _sample_pool(domain, [small], [small, large], enabled=False)
    with_sample = _sample_pool(domain, [small], [small, large], enabled=True)

    assert not without.sample_states
    assert with_sample.sample_states
    assert SAMPLE_ONLY_CONCEPT not in without.concepts
    assert SAMPLE_ONLY_CONCEPT in with_sample.concepts
    assert set(without.concepts) < set(with_sample.concepts)

    # ... and it is dropped *because* it coincides on the training states: some other concept
    # of the pool has the same denotation there, and differs once the sample is included.
    target = with_sample.concepts[SAMPLE_ONLY_CONCEPT]
    states = with_sample._dedup_states()
    training_states = states[: len(states) - len(with_sample.sample_states)]
    target_on_training = _denotations(target, training_states)
    twins = {
        name: concept
        for name, concept in with_sample.concepts.items()
        if name != SAMPLE_ONLY_CONCEPT and _denotations(concept, training_states) == target_on_training
    }
    assert twins, "the concept should be indistinguishable from another one on the training states"
    target_everywhere = _denotations(target, states)
    assert all(_denotations(twin, states) != target_everywhere for twin in twins.values())


def test_redundant_concept_pruning_uses_the_sample(blocks3ops_small, blocks3ops):
    """compute_redundant_concepts separates two concepts the training states cannot separate.

    Preset concepts bypass dlplan's own deduplication, so this exercises genfond's pruning
    directly: the two below have the same extension on every state of the 3-block instance.
    """
    domain, small = blocks3ops_small
    _, large = blocks3ops
    presets = {"concepts": [SAMPLE_ONLY_CONCEPT, "c_not(c_primitive(ontable_G,0))"], "booleans": [], "roles": []}
    without = _sample_pool(domain, [small], [small, large], enabled=False, preset_features=presets)
    with_sample = _sample_pool(domain, [small], [small, large], enabled=True, preset_features=presets)

    assert len(without.compute_redundant_concepts()) == 1
    assert without.compute_redundant_concepts() < set(without.concepts)
    assert with_sample.compute_redundant_concepts() == set()


def test_feature_sample_leaves_the_asp_instance_alone(blocks3ops_small, blocks3ops):
    """Sample states never become state/2 facts."""
    domain, small = blocks3ops_small
    _, large = blocks3ops
    with_sample = _sample_pool(domain, [small], [small, large], enabled=True)
    instance = with_sample.to_clingo()
    num_states = sum(len(graph.nodes) for graph in with_sample.state_graphs.values())
    assert instance.count("state(") - instance.count("aug_state(") == num_states
    # The extra instance built for the sample-only problem is not addressable from the ASP.
    assert with_sample.sample_instances
    assert set(with_sample.sample_instances) & set(with_sample.problem_name_to_id) == set()
