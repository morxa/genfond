import yaml

DEFAULT_CONFIG = {
    'min_complexity': 2,
    'max_complexity': 9,
    'policy_iterations': 100,
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
}

DEFAULT_TYPE_CONFIGS = {
    'datalog': {
        'include_numerical_features': False,
        'include_concepts': True,
        'include_roles': True,
    },
}


class ConfigHandler(dict):

    def __init__(self, config_file_object=None, type=None):
        self.update(DEFAULT_CONFIG)
        if type and type in DEFAULT_TYPE_CONFIGS:
            self.update(DEFAULT_TYPE_CONFIGS[type])
        if config_file_object:
            self.update(yaml.safe_load(config_file_object))
