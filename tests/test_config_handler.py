from genfond.config_handler import ConfigHandler


def test_config_handler(simple_config):
    config = ConfigHandler(simple_config)
    assert config['min_complexity'] == 3
    assert config['max_complexity'] == 15
    assert config['keep_going'] is True
    assert config['num_threads'] == 8
    assert config['include_boolean_features'] is True
    assert config['include_numerical_features'] is True
    assert config['include_concepts'] is False
    assert config['include_roles'] is False


def test_datalog_config_handler():
    config = ConfigHandler(type='datalog')
    assert config['include_numerical_features'] is False
    assert config['include_concepts'] is True
    assert config['include_roles'] is True


def test_config_handler_override(simple_config):
    config = ConfigHandler(simple_config, override={'min_complexity': 5, 'num_threads': None})
    assert config['min_complexity'] == 5
    # Still using the config value
    assert config['num_threads'] == 8
