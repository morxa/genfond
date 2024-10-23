from genfond.datalog_policy import DatalogPolicyRule, DatalogPolicy, Cond
from genfond.execute_datalog_policy import execute_datalog_policy
from genfond.config_handler import ConfigHandler
import pytest


def test_block_clear_all(blocks_clear):
    domain, problem = blocks_clear

    policy = DatalogPolicy([
        DatalogPolicyRule('unstack(X, Y)', concepts=[
            ('X', 'c_primitive(clear, 0)'),
        ]),
        DatalogPolicyRule('putdown(X)', concepts=[
            ('X', 'c_primitive(clear, 0)'),
        ]),
    ])
    config = ConfigHandler()
    execute_datalog_policy(domain, problem, policy, config)


def test_fond_blocks(fond_blocks):
    domain, problem = fond_blocks

    policy = DatalogPolicy([
        DatalogPolicyRule(
            'puton(X, Y, Z)',
            concepts=[
                ('Y', 'c_one_of(Table)'),
                ('X',
                 'c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),c_projection(r_primitive(on_G,0,1),0))'),
            ]),
        DatalogPolicyRule(
            'puton(X, Y, Z)',
            concepts=[
                ('Y', 'c_one_of(Table)'),
                ('X',
                 'c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),c_projection(r_primitive(on_G,0,1),1))'),
            ]),
        DatalogPolicyRule('puton(X, Y, Z)',
                          concepts=[
                              ('X', 'c_projection(r_primitive(on_G,0,1),0)'),
                              ('Y', 'c_projection(r_primitive(on_G,0,1),1)'),
                          ]),
    ])
    config = ConfigHandler()
    execute_datalog_policy(domain, problem, policy, config)


def test_datalog_policy_with_conds(doors):
    domain, problem = doors
    policy = DatalogPolicy([
        DatalogPolicyRule('move-forward-door-open(P1, P2, P3, P4)', conds={'b_nullary(hold-key)': Cond.TRUE}),
        DatalogPolicyRule('move-forward-door-closed(P1, P2, P3, P4)', conds={'b_nullary(hold-key)': Cond.TRUE}),
        DatalogPolicyRule('move-forward-last-door-open(P1, P2, P3)', conds={'b_nullary(hold-key)': Cond.TRUE}),
        DatalogPolicyRule('move-forward-last-door-closed(P1, P2, P3)', conds={'b_nullary(hold-key)': Cond.TRUE}),
        DatalogPolicyRule('pick-key(X)', conds={'b_nullary(hold-key)': Cond.FALSE}),
    ])
    config = ConfigHandler()
    config['policy_steps'] = 10
    for _ in range(10):
        execute_datalog_policy(domain, problem, policy, config)


def test_datalog_policy_with_roles(fond_blocks):
    domain, problem = fond_blocks
    policy = DatalogPolicy([
        DatalogPolicyRule(
            'puton(X, Y, Z)',
            roles=[
                ('X', 'Y', 'r_primitive(on_G,0,1)'),
            ],
        ),
        DatalogPolicyRule(
            'puton(X, Y, Z)',
            concepts=[('Y', 'c_one_of(Table)')],
        ),
    ])
    config = ConfigHandler()
    config['policy_steps'] = 10
    for _ in range(10):
        execute_datalog_policy(domain, problem, policy, config)


def test_blocks3ops(blocks3ops):
    domain, problem = blocks3ops

    cond1 = 'b_empty(r_restrict(r_primitive(on, 0, 1), c_and(c_primitive(ontable, 0), c_not(c_primitive(ontable_G, 0)))))'
    cond2 = 'b_empty(r_and(r_primitive(on, 0, 1), r_not(r_primitive(on_G, 0, 1))))'

    policy = DatalogPolicy([
        DatalogPolicyRule(
            'newtower(X, Y)',
            conds={
                cond1: Cond.FALSE,
            },
        ),
        DatalogPolicyRule(
            'newtower(X, Y)',
            conds={
                cond2: Cond.FALSE,
            },
        ),
        DatalogPolicyRule('stack(X, Y)',
                          concepts=[
                              ('Y', 'c_primitive(ontable, 0)'),
                              ('Y', 'c_primitive(ontable_G, 0)'),
                          ],
                          roles=[('X', 'Y', 'r_primitive(on_G, 0, 1)')],
                          conds={
                              cond1: Cond.TRUE,
                              cond2: Cond.TRUE,
                          }),
        DatalogPolicyRule('stack(X, Y)',
                          concepts=[
                              ('Y', 'c_some(r_and(r_primitive(on, 0, 1), r_inverse(r_primitive(on, 1, 0))), c_top)'),
                          ],
                          roles=[('X', 'Y', 'r_primitive(on_G, 0, 1)')],
                          conds={
                              cond1: Cond.TRUE,
                              cond2: Cond.TRUE,
                          }),
    ])
    config = ConfigHandler()
    config['policy_steps'] = 10
    execute_datalog_policy(domain, problem, policy, config)


def test_execute_datalog_policy_with_augmented_states(blocks_clear):
    domain, problem = blocks_clear
    policy = DatalogPolicy([
        DatalogPolicyRule(
            'unstack(X, Y)',
            state_aug_conds={'b_empty(c_and(c_primitive(clear_G,0),c_primitive(aparam1,0)))': Cond.FALSE})
    ])
    config = ConfigHandler()
    config['include_actions'] = True
    config['policy_steps'] = 1
    execute_datalog_policy(domain, problem, policy, config)
    policy = DatalogPolicy([
        DatalogPolicyRule(
            'unstack(X, Y)',
            state_aug_conds={'b_empty(c_and(c_primitive(clear_G,0),c_primitive(aparam0,0)))': Cond.FALSE})
    ])
    with pytest.raises(RuntimeError):
        execute_datalog_policy(domain, problem, policy, config)
