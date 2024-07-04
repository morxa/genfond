from genfond.generate_datalog_policy import generate_datalog_policy
from genfond.execute_datalog_policy import execute_datalog_policy
from genfond.solver import Solver
from genfond.datalog_policy import DatalogPolicy, DatalogPolicyRule, Cond


def test_generate_datalog_policy():
    # A  B  E
    # C  D  F
    # goal: on(C, D)
    solution = {
        'cost': 5,
        'c_selected': {
            'c_one_of(Table)',
            'c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),c_projection(r_primitive(on_G,0,1),0))',
            'c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),c_projection(r_primitive(on_G,0,1),1))',
            'c_projection(r_primitive(on_G,0,1),0)', 'c_projection(r_primitive(on_G,0,1),1)'
        },
        'good_action': {
            (0, 0, 'puton(A, Table, C)'),
            (0, 1, 'puton(B, Table, D)'),
            (0, 2, 'puton(C, D, Table)'),
        },
        'c_distinguished': {
            (0, 0, 'puton(A, Table, C)', 0, 0, 'puton(A, B, C)', 'c_one_of(Table)', 'pos', 2),
            (0, 0, 'puton(A, Table, C)', 0, 0, 'puton(A, E, C)', 'c_one_of(Table)', 'pos', 2),
            (0, 0, 'puton(A, Table, C)', 0, 0, 'puton(E, Table, F)',
             'c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),c_projection(r_primitive(on_G,0,1),0))',
             'pos', 1),
            (0, 1, 'puton(B, Table, D)', 0, 1, 'puton(A, Table, C)', 'c_one_of(Table)', 'pos', 2),
            (0, 1, 'puton(B, Table, D)', 0, 1, 'puton(E, Table, F)',
             'c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),c_projection(r_primitive(on_G,0,1),1))',
             'pos', 1),
            (0, 2, 'puton(C, D, Table)', 0, 2, 'puton(D, C, Table)', 'c_projection(r_primitive(on_G,0,1),0)', 'pos',
             1),
            (0, 2, 'puton(C, D, Table)', 0, 2, 'puton(D, C, Table)', 'c_projection(r_primitive(on_G,0,1),1)', 'pos',
             2),
        },
    }
    policy = generate_datalog_policy(solution)
    print(policy)
    assert policy == DatalogPolicy([
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


def test_generate_datalog_policy_with_conds():
    solution = {
        'cost': 2,
        'f_selected': {
            'b_nullary(hold-key)',
        },
        'good_action': {
            (0, 0, 'pick-key(k1)'),
            (0, 1, 'move-forward-door-open(l1, l2, d1, d2)'),
            (0, 2, 'move-forward-door-open(l2, l3, d2, d3)'),
            (0, 3, 'move-forward-last-door-closed(l3, l4, d3)'),
        },
        'f_distinguished': {
            (0, 0, 0, 1, 'b_nullary(hold-key)'),
            (0, 1, 0, 0, 'b_nullary(hold-key)'),
            (0, 2, 0, 0, 'b_nullary(hold-key)'),
            (0, 3, 0, 0, 'b_nullary(hold-key)'),
        },
        'bool_eval': {
            (0, 0, 'b_nullary(hold-key)', 0),
            (0, 1, 'b_nullary(hold-key)', 1),
            (0, 2, 'b_nullary(hold-key)', 1),
            (0, 3, 'b_nullary(hold-key)', 1),
        },
    }
    policy = generate_datalog_policy(solution)
    print(policy)
    assert policy == DatalogPolicy([
        DatalogPolicyRule('move-forward-door-open(P1, P2, P3, P4)', conds={'b_nullary(hold-key)': Cond.TRUE}),
        DatalogPolicyRule('move-forward-last-door-closed(P1, P2, P3)', conds={'b_nullary(hold-key)': Cond.TRUE}),
        DatalogPolicyRule('pick-key(X)', conds={'b_nullary(hold-key)': Cond.FALSE}),
    ])


def test_generate_datalog_policy_with_role_conds():
    solution = {
        'cost': 2,
        'r_selected': {'r_primitive(on_G,0,1)'},
        'good_action': {
            (0, 0, 'puton(A, C, Table)'),
        },
        'r_distinguished': {
            (0, 0, 'puton(A, C, Table)', 0, 0, 'puton(A, B, C)', 'r_primitive(on_G,0,1)', 'pos', 1, 2),
        },
    }
    policy = generate_datalog_policy(solution)
    print(policy)
    assert policy == DatalogPolicy([
        DatalogPolicyRule('puton(P, Q, R)', roles=[('P', 'Q', 'r_primitive(on_G,0,1)')]),
    ])
