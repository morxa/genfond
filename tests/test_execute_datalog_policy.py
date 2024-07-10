from genfond.datalog_policy import DatalogPolicyRule, DatalogPolicy, Cond
from genfond.execute_datalog_policy import execute_datalog_policy


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

    execute_datalog_policy(domain, problem, policy)


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

    execute_datalog_policy(domain, problem, policy)


def test_datalog_policy_with_conds(doors):
    domain, problem = doors
    policy = DatalogPolicy([
        DatalogPolicyRule('move-forward-door-open(P1, P2, P3, P4)', conds={'b_nullary(hold-key)': Cond.TRUE}),
        DatalogPolicyRule('move-forward-door-closed(P1, P2, P3, P4)', conds={'b_nullary(hold-key)': Cond.TRUE}),
        DatalogPolicyRule('move-forward-last-door-open(P1, P2, P3)', conds={'b_nullary(hold-key)': Cond.TRUE}),
        DatalogPolicyRule('move-forward-last-door-closed(P1, P2, P3)', conds={'b_nullary(hold-key)': Cond.TRUE}),
        DatalogPolicyRule('pick-key(X)', conds={'b_nullary(hold-key)': Cond.FALSE}),
    ])
    for _ in range(10):
        execute_datalog_policy(domain, problem, policy, max_steps=10)


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
    for _ in range(10):
        execute_datalog_policy(domain, problem, policy, max_steps=10)
