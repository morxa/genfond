from genfond.datalog_policy import DatalogPolicyRule, DatalogPolicy
from genfond.execute_datalog_policy import execute_datalog_policy


def test_block_clear_all(blocks_clear):
    domain, problem = blocks_clear
    
    policy = DatalogPolicy([
        DatalogPolicyRule('unstack(X, Y)', [
            ('X', 'c_primitive(clear, 0)'),
        ]),
        DatalogPolicyRule('putdown(X)', [
            ('X', 'c_primitive(clear, 0)'),
        ]),
    ])
    
    execute_datalog_policy(domain, problem, policy, max_steps=0)
    
def test_fond_blocks(fond_blocks):
    domain, problem = fond_blocks
    
    policy = DatalogPolicy([
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
    
    execute_datalog_policy(domain, problem, policy, max_steps=0)
