import yaml
import mergedeep

DEFAULT_CONFIG = {
    'min_complexity': 2,
    'max_complexity': 9,
    'policy_iterations': 100,
    'abort_on_cycle': False,
    'policy_steps': 10000,
    'num_threads': None,
    'max_memory': None,
    'dump_failed_policies': False,
    'keep_going': False,
    'continue_after_error': False,
    'include_boolean_features': True,
    'include_numerical_features': True,
    'include_concepts': False,
    'include_roles': False,
    'include_actions': False,
    'include_action_params': False,
    'include_pristine_states': True,
    'include_dead_states': True,
    'preset_features': None,
    'feature_generator': {},
    'prune_roles': True,
    'prune_concepts': True,
    'prune_features': False,
    'use_unrestricted_features': True,
    'feature_generator': {
        'generate_til_c_role': False,
    },
    'unrestricted_feature_generator': {
        'generate_empty_boolean': True,
        'generate_inclusion_boolean': True,
        'generate_nullary_boolean': True,
        'generate_all_concept': True,
        'generate_and_concept': True,
        'generate_bot_concept': True,
        'generate_diff_concept': True,
        'generate_equal_concept': True,
        'generate_not_concept': True,
        'generate_one_of_concept': True,
        'generate_or_concept': True,
        'generate_primitive_concept': True,
        'generate_projection_concept': True,
        'generate_some_concept': True,
        'generate_subset_concept': True,
        'generate_top_concept': True,
        'generate_concept_distance_numerical': True,
        'generate_count_numerical': True,
        'generate_and_role': True,
        'generate_compose_role': True,
        'generate_diff_role': True,
        'generate_identity_role': True,
        'generate_inverse_role': True,
        'generate_not_role': True,
        'generate_or_role': True,
        'generate_primitive_role': True,
        'generate_restrict_role': True,
        'generate_til_c_role': True,
        'generate_top_role': True,
        'generate_transitive_closure_role': True,
        'generate_transitive_reflexive_closure_role': True,
    },
    'log': {
        'execution': 'CRITICAL',
    },
}

DEFAULT_TYPE_CONFIGS = {
    'datalog': {
        'policy_type': 'DATALOG',
        'solve_prog': 'solve_datalog.lp',
        'include_numerical_features': False,
        'include_concepts': True,
        'include_roles': True,
        'include_actions': False,
        'include_dead_states': False,
        'feature_generator': {
            'generate_concept_distance_numerical': False,
            'generate_count_numerical': False,
            'generate_til_c_role': True,
        }
    },
    'datalog-action-params': {
        'policy_type': 'DATALOG',
        'solve_prog': 'solve_datalog_action_params.lp',
        'include_action_params': True,
        'include_roles': False,
        'include_concepts': False,
    },
    'datalog-actions': {
        'policy_type': 'DATALOG',
        'solve_prog': 'solve_datalog_actions.lp',
        'include_pristine_states': False,
        'include_actions': True,
        'include_roles': False,
        'include_concepts': False,
        'include_action_params': False,
    },
    'exact': {
        'policy_type': 'EXACT',
        'solve_prog': 'solve.lp',
    },
    'state': {
        'policy_type': 'CONSTRAINED',
        'solve_prog': 'solve_state_constraints.lp',
    },
    'trans': {
        'policy_type': 'CONSTRAINED',
        'solve_prog': 'solve_trans_constraints.lp',
    },
}


class ConfigHandler(dict):

    def __init__(self, config_file_object=None, type=None, override=None):
        mergedeep.merge(self, DEFAULT_CONFIG)
        if type and type in DEFAULT_TYPE_CONFIGS:
            mergedeep.merge(self, DEFAULT_TYPE_CONFIGS[type])
        if config_file_object:
            mergedeep.merge(self, yaml.safe_load(config_file_object))
        if override:
            # Only override values that are already in the config that have been set to a non-None value
            mergedeep.merge(self, {k: v for k, v in override.items() if k in self and v is not None})

    def dump(self):
        dump = dict()
        mergedeep.merge(dump, self)
        return yaml.dump(dump, default_flow_style=False)
