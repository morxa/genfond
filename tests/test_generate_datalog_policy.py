from genfond.generate_datalog_policy import generate_datalog_policy
from genfond.execute_datalog_policy import execute_datalog_policy
from genfond.solver import Solver
from genfond.datalog_policy import DatalogPolicy, DatalogPolicyRule


def test_generate_datalog_policy():
    # A  B  E
    # C  D  F
    # goal: on(C, D)
    solution = {
        'cost': 5,
        'selected': {
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
        'distinguished': {
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
        DatalogPolicyRule('puton(X, Y, Z)', [
            ('Y', 'c_one_of(Table)'),
            ('X', 'c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),c_projection(r_primitive(on_G,0,1),0))'),
        ]),
        DatalogPolicyRule('puton(X, Y, Z)', [
            ('Y', 'c_one_of(Table)'),
            ('X', 'c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),c_projection(r_primitive(on_G,0,1),1))'),
        ]),
        DatalogPolicyRule('puton(X, Y, Z)', [
            ('X', 'c_projection(r_primitive(on_G,0,1),0)'),
            ('Y', 'c_projection(r_primitive(on_G,0,1),1)'),
        ]),
    ])
