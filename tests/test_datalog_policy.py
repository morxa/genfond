from genfond.datalog_policy import DatalogPolicy, DatalogPolicyRule


def test_datalog_policy_equals():    
    # EQC1: policy1, policy2
    # EQC2: policy3, policy4
    
    policy1 = DatalogPolicy([
        DatalogPolicyRule('p(X, Y)', [
            ('X', 'c_primitive(is_a, 0)'),
            ('Y', 'c_primitive(is_b, 0)'),
        ]),
        DatalogPolicyRule('q(X)', [
            ('X', 'c_primitive(is_a, 0)'),
        ]),
    ])
    
    policy2 = DatalogPolicy([
        DatalogPolicyRule('q(X)', [
            ('X', 'c_primitive(is_a, 0)'),
        ]),
        DatalogPolicyRule('q(A)', [
            ('A', 'c_primitive(is_a, 0)'),
            ('A', 'c_primitive(is_a, 0)'),
        ]),
        DatalogPolicyRule('p(Y, X)', [
            ('X', 'c_primitive(is_b, 0)'),
            ('Y', 'c_primitive(is_a, 0)'),
        ]),
    ])
    
    policy3 = DatalogPolicy([
        DatalogPolicyRule('q(X)', [
            ('X', 'c_primitive(is_b, 0)'),
        ]),
        DatalogPolicyRule('p(X, Y)', [
            ('Y', 'c_primitive(is_b, 0)'),
            ('X', 'c_primitive(is_a, 0)'),
        ]),
    ])
    
    policy4 = DatalogPolicy([
        DatalogPolicyRule('q(X)', [
            ('X', 'c_primitive(is_b, 0)'),
        ]),
        DatalogPolicyRule('p(X, Y)', [
            ('Y', 'c_primitive(is_b, 0)'),
            ('X', 'c_primitive(is_a, 0)'),
        ]),
        DatalogPolicyRule('q(X)', [
            ('X', 'c_primitive(is_b, 0)'),
        ]),
    ])    
    
    assert policy1 == policy1
    assert policy1 == policy2
    assert policy1 != policy3
    assert policy1 != policy4
    
    assert policy2 == policy1
    assert policy2 == policy2
    assert policy2 != policy3
    assert policy2 != policy4
    
    assert policy3 != policy1
    assert policy3 != policy2
    assert policy3 == policy3
    assert policy3 == policy4
    
    assert policy4 != policy1
    assert policy4 != policy2
    assert policy4 == policy3
    assert policy4 == policy4
